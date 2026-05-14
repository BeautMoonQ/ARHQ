# 当前 ARHQ 实现整理

本文基于当前代码整理 ARHQ / SVDQuant / R-only 的完整流程，包括校准数据、分解、smoothing、NVFP4 模拟量化、推理替换逻辑和评测方式。

## 范围

当前流程针对 `Qwen3-4B-Thinking-2507`，通过把选定的 `nn.Linear` 替换成 PyTorch fake-quant 模块，做模拟 NVFP4 推理。

支持的模块集合定义在 `arhq/eval_quantized.py`：

- `attn`: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `ffn`: `gate_proj`, `up_proj`, `down_proj`
- `all`: 上面 7 个 projection 全部替换

模型共有 36 层。对 `module_set=all`、`rank=128` 来说，完整分解覆盖应包括：

- 每个 attention projection 和 FFN projection 都有 36 个 layer 文件
- 当前主要使用 3 组配置：`r_only raw`、`r_only smoothing`、`svdquant smoothing`
- all-layer rank-128 的完整文件数是 `36 * 7 * 3 = 756`

当前本地 `results/layer_results/layer_*` 中，这 7 个 projection 的上述 3 组配置都已经覆盖 36 层。

## 校准数据

校准数据由 `reference_code/hadazca_calib.py` 收集。

脚本读取已有推理结果目录，重建 prefill token 序列：

```text
chat_template(question) + output_ids
```

然后对目标 linear 注册 forward hook，保存这些 linear 的输入 activation。

关键参数：

- `--module_set attn|ffn|all|custom`
- `--max_layer_tokens`：每个目标 linear 最多收集多少 token；当前常用 `30000`
- `--save_tag`：用于区分保存目录，例如 FFN 数据用 `vllm_ZebraLogic_ffn`
- `--repeat_index 0`：通常使用 vLLM 标准输出的 repeat 0

保存目录结构：

```text
{calib_dir}/{precision}_{dataset}[_save_tag]/samples_{start}/layer_{i}/
  activations.pt
  activations_truncated.pt
  weights.pt
```

当前 ZebraLogic 校准数据目录：

```text
/home/wangyifeng/work/data/calib_zebralogic/vllm_ZebraLogic/samples_0000
/home/wangyifeng/work/data/calib_zebralogic/vllm_ZebraLogic_ffn/samples_0000
```

典型 layer 0 shape：

| Projection | Activation shape | Weight shape |
|---|---:|---:|
| `q_proj` | `(30000, 2560)` | `(4096, 2560)` |
| `k_proj` | `(30000, 2560)` | `(1024, 2560)` |
| `v_proj` | `(30000, 2560)` | `(1024, 2560)` |
| `o_proj` | `(30000, 4096)` | `(2560, 4096)` |
| `gate_proj` | `(30000, 2560)` | `(9728, 2560)` |
| `up_proj` | `(30000, 2560)` | `(9728, 2560)` |
| `down_proj` | `(30000, 9728)` | `(2560, 9728)` |

启动脚本：

```bash
./run_zebralogic_calib.sh cuda:0 0-127 0-35 30000
./run_zebralogic_calib_ffn.sh cuda:0 0-127 0-35 30000
```

## NVFP4 模拟量化

fake quant 实现在 `arhq/quant.py`。

推理主路径使用：

```python
nvfp4_quantize(x, block_size=16)
```

这是返回 dequantized tensor 的模拟量化，不是 packed low-bit kernel。

流程：

1. 把最后一维 pad 到 16 的倍数。
2. reshape 成 `[-1, 16]` block。
3. 每 16 个值计算一个 scale：`amax / 6`。
4. 对 scale 做 FP8 E4M3 精度模拟。
5. 把值 round 到 FP4 E2M1 codebook：

```text
0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6
```

6. 乘回 scale，返回 dequantized tensor。

实现是向量化的。forward 中没有 Python for 循环，但 `_fp4_round` 会构造 `[..., 15]` 的 diff tensor 并做 `argmin`，长生成时开销明显。

`arhq/quant.py` 里还有 `nvfp4_quantize_2d(W, block_h=16, block_w=16)`，但当前 `arhq/eval_quantized.py` 推理路径没有使用它。当前推理中 activation 和 residual weight 都使用 1D block 量化。

## Smoothing

smoothing 实现在 `arhq/transforms.py`。

对 activation `A: [N, D_in]` 和权重 `W: [D_out, D_in]`，per-channel scale 为：

```python
act_max = A.abs().amax(dim=0)
w_max = W.abs().amax(dim=0)
scale = act_max.pow(alpha) / w_max.pow(1 - alpha)
```

变换为：

```text
A_s = A / scale
W_s = W * scale
```

`search_best_alpha` 会扫描：

```text
alpha = 0.00, 0.05, ..., 1.00
```

选择使下面 MSE 最小的 alpha：

```text
Q(A / scale) @ Q(W * scale).T
```

目标是逼近：

```text
A @ W.T
```

smoothing 分解文件会额外保存：

```python
{
  "scale": scale.half().cpu(),
  "alpha": best_alpha,
}
```

## 低秩分解方法

分解代码在 `arhq/lowrank.py`。

所有方法最终都会生成：

```python
{
  "B_r":   [D_out, rank],
  "A_fac": [D_in, rank],
  "W_res": [D_out, D_in],
}
```

低秩分支表示：

```text
W_sig = B_r @ A_fac.T
```

### SVDQuant

`svdquant_decompose(W, rank)` 对 `W` 做截断 SVD：

```text
W ~= B_r @ A_fac.T
W_res = W - B_r @ A_fac.T
```

它不使用校准 activation。

### ARHQ

`arhq_decompose(W, A_calib, rank)` 构造 activation-residual weighted Hessian：

```text
H   = A_calib.T @ A_calib / N
E_a = A_calib - Q(A_calib)
R_a = E_a.T @ E_a / N
beta_a = mean(diag(H)) / mean(diag(R_a))
H_res = H + beta_a * R_a
```

然后在 `H_res` 度量下做分解：

```text
M = W @ H_res^{1/2}
M ~= B_r @ A_tilde.T
A_fac = H_res^{-1/2} @ A_tilde
W_res = W - B_r @ A_fac.T
```

函数签名里支持可选的 weight quant residual 项，但当前 sweep 路径没有传 `w_quant_fn`，所以当前实际使用的是只包含 activation residual 的 ARHQ。

### R-only

`r_only_decompose(W, A_calib, rank)` 只使用 activation quantization residual covariance：

```text
E_a = A_calib - Q(A_calib)
R_a = E_a.T @ E_a / N
```

然后：

```text
M = W @ R_a^{1/2}
M ~= B_r @ A_tilde.T
A_fac = R_a^{-1/2} @ A_tilde
W_res = W - B_r @ A_fac.T
```

R-only 不使用 activation Hessian `H`，也没有 beta scaling。

## Raw 与 Smoothing 分解区别

`arhq/lowrank.py::evaluate_single` 是共同的分解和离线评估入口。

`raw` 设置下：

```text
A_t = A
W_t = W
```

`smoothing` 设置下：

```text
A_t = A / scale
W_t = W * scale
```

分解始终在 `W_t` 和 `A_calib_t` 上执行。

离线 SNR 评估使用：

```text
Y_true = A_eval @ W.T
Y_main = Q(A_eval_t) @ Q(W_res).T
Y_lr = (A_eval_t @ A_fac) @ B_r.T
Y_hat = Y_main + Y_lr
```

因此 smoothing 的目标仍然是原始空间的 `A @ W.T`，只是把量化难度在 activation 和 weight 之间重新分配。

## 分解 sweep 与保存文件

Attention sweep：

```bash
./run_three_ronly_zebralogic.sh cuda:0 0-35 128 \
  /home/wangyifeng/work/data/calib_zebralogic/vllm_ZebraLogic/samples_0000 \
  zebralogic
```

它运行 `arhq/sweep_three_ronly.py`，配置为：

```python
CONFIGS = [
    ("r_only", "raw"),
    ("r_only", "smoothing"),
    ("svdquant", "smoothing"),
]
```

FFN sweep：

```bash
./run_three_ronly_zebralogic_ffn.sh cuda:1 128 0-35 \
  /home/wangyifeng/work/data/calib_zebralogic/vllm_ZebraLogic_ffn/samples_0000 \
  zebralogic 2048 gate_proj,up_proj,down_proj
```

它运行 `arhq/sweep_three_ronly_ffn.py`，同样跑上述 3 组配置。

分解参数保存为：

```text
results/layer_results/layer_{layer_idx}/{proj}_{method}_{setting}_rank{rank}.pt
```

例如：

```text
results/layer_results/layer_0/q_proj_r_only_smoothing_rank128.pt
results/layer_results/layer_0/gate_proj_svdquant_smoothing_rank128.pt
```

每个文件包含：

```python
{
  "B_r": ...,
  "A_fac": ...,
  "W_res": ...,
  "rank": 128,
  "method": "r_only" | "svdquant" | "arhq",
  "setting": "raw" | "smoothing",
  "beta": ...,
  # 仅 smoothing 有：
  "scale": ...,
  "alpha": ...,
}
```

注意：当前推理阶段不再直接信任文件里的 `W_res`。推理只使用保存的 `B_r`、`A_fac` 和可选 `scale`，然后根据当前加载模型的权重重新计算 `W_res`。

## 推理阶段的模块替换

主入口是 `arhq/eval_quantized.py`。

模型加载方式：

```python
AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True,
)
```

除了 `original` 方法外，都会调用 `replace_linear_projections` 替换选中的 linear。

### Baseline

`--method baseline` 时，每个选中的 linear 会被替换成 `QuantizedOnlyLinear`。

初始化：

```text
W_q = Q(W_orig)
```

forward：

```text
Y = Q(X) @ W_q.T
```

baseline 没有低秩分支，也不使用 smoothing。

### 低秩方法

`--method r_only|svdquant|arhq` 时，每个 projection 会加载：

```text
{proj}_{method}_{setting}_rank{rank}.pt
```

如果文件缺失，该 projection 会 fallback 到 baseline fake quant。

runtime residual 重建由 `build_runtime_residual` 完成。

Raw：

```text
W_target = W_orig
W_res = W_orig - B_r @ A_fac.T
```

Smoothing：

```text
W_target = W_orig * scale
W_res = W_orig * scale - B_r @ A_fac.T
```

这样做是为了避免校准阶段保存的 `weights.pt` 和当前推理加载的模型 checkpoint 之间存在细微差异。

替换后的模块是 `QuantizedLowRankLinear`。

初始化：

```text
W_res_q = Q(W_res)
```

`W_res_q` 会作为 buffer 保存。未量化的 `W_res` 随后被清空，以降低 all-layer FFN 推理时的显存压力。

Raw forward：

```text
X_s = X
Y = Q(X_s) @ W_res_q.T + (X_s @ A_fac) @ B_r.T
```

Smoothing forward：

```text
X_s = X / scale
Y = Q(X_s) @ W_res_q.T + (X_s @ A_fac) @ B_r.T
```

主 residual 分支是 fake quant。低秩分支不量化，使用模型 compute dtype，目前该脚本里是 fp16。

## ZebraLogic All-Layer 评测启动方式

All-layer R-only smoothing：

```bash
./run_eval_zebralogic_ronly_smoothing_all.sh cuda:0 4 1
```

核心命令等价于：

```bash
conda run -n llmc --no-capture-output python -m arhq.eval_quantized \
  --method r_only --setting smoothing --rank 128 --device cuda:0 \
  --module_set all \
  --datasets ZebraLogic --batch_size 4 --repeat_num 1 \
  --max_new_tokens 52768 --temperature 0.6 --top_p 0.95 --top_k 20 \
  --think_mode think
```

All-layer SVDQuant smoothing：

```bash
./run_eval_zebralogic_svdq_smoothing_all.sh cuda:0 4 1
```

All-layer R-only raw：

```bash
./run_eval_zebralogic_ronly_raw_all.sh cuda:0 4 1
```

通用脚本：

```bash
./run.sh r_only 128 ZebraLogic cuda:0 smoothing all
./run.sh svdquant 128 ZebraLogic cuda:0 smoothing all
./run.sh r_only 128 ZebraLogic cuda:0 raw all
```

默认输出目录名：

- `arhq_r_only_smoothing_rank128_all`
- `arhq_svdquant_smoothing_rank128_all`
- `arhq_r_only_raw_rank128_all`
- `arhq_baseline`
- `arhq_original`

## 评测

ZebraLogic 应使用 `calc_acc.py` 中的 ZebraLogic 专用逻辑评测。

原因：`reference_code/eval_utils.py::extract_answer` 当前把 `ZebraLogic` 当作选择题处理，会调用 `_extract_letter`，因此即使 `content` 中有正确 JSON 表格，`extracted_answer` 也可能是 `null`。

`calc_acc.py::work_zebralogic` 会直接解析 `content`，归一化表格后比较。

指标含义：

- `Puzzle-level accuracy`：整张表完全正确
- `Cell-level accuracy`：正确 aligned cells / aligned cells
- `Cell precision`：正确 cells / 预测 cells
- `Cell recall`：正确 cells / gold cells
- `Cell F1`：cell precision 和 cell recall 的调和平均

解析器支持标准表格格式：

```json
{"header": [...], "rows": [[...], ...]}
```

也支持 house-key JSON：

```json
{
  "1": {"name": "...", "nationality": "..."},
  "2": {"name": "...", "nationality": "..."}
}
```

已知严格匹配问题：一些语义等价别名仍会被判错，例如 `apr` vs `april`。

最近一次本地 all-layer R-only smoothing ZebraLogic 聚合检查：

- 48 条严格代码评测：`45/48 = 93.75%`
- 人工语义评测，允许 `apr == april`：`46/48 = 95.83%`
- 明确错误记录：`data[2]`、`data[34]`

## 一致性检查

使用 `tools/check_inference_decomp_consistency.py` 检查保存的分解文件是否和当前模型权重一致。

Raw 检查：

```text
W_orig ~= W_res + B_r @ A_fac.T
```

Smoothing 检查：

```text
W_orig * scale ~= W_res + B_r @ A_fac.T
```

脚本也会报告推理实际使用的近似，也就是 residual weight 量化之后：

```text
W_target ~= Q(W_res) + B_r @ A_fac.T
```

示例：

```bash
conda run -n llmc --no-capture-output python tools/check_inference_decomp_consistency.py \
  --method r_only --setting smoothing --rank 128 \
  --module_set all --layers 0-35
```

## 重要限制

- 当前是模拟量化，不是生产级 packed NVFP4 kernel。
- `nvfp4_quantize` 返回 dequantized tensor，所以显存和速度都不等价于真实 NVFP4 部署。
- activation 每次 forward 都会动态量化，包括 decode 阶段每一步。
- residual weight 在模块构造时预量化一次，避免每次 forward 重复量化 weight。
- 低秩分支不量化，仍是 dense fp16 matmul。
- all-layer 同时替换 attention 和 FFN，会引入很多 fake-quant 调用，明显慢于 attention-only。
- `eval_quantized.py` 内部 batch accuracy 对 ZebraLogic 不可靠，因为它直接比较 `extracted_answer`；最终准确率应以 `calc_acc.py` 为准。

