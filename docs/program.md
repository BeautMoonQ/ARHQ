# FP4 Rotation Quantization — Layer 2 Autoresearch

## 目标

找到针对 **Layer 2** 的最优激活变换方案，使 NVFP4 量化后的输出误差最小化。

主指标：`avg_snr_improvement_db`（相对 no-transform baseline 的 SNR 改善，越高越好）

---

## 数据与评估配置

- **评估层**：Layer 2（`EVAL_LAYERS = [2]`，在 prepare.py 中固定）
- **数据来源**：`~/work/data/calib/vllm_MATH-500/samples_0000/layer_2/`
  - 校准集：前 2048 tokens（`load_calibration_data()` 返回）
  - 评估集：接下来的 2048 tokens（`load_eval_data()` 返回）
  - 两者不重叠，均来自同一组 16 个 MATH-500 样本
- **量化**：NVFP4 E2M1，per-block（block_size=16）FP8 E4M3 scale
- **维度**：所有 proj 输入 dim = 2560
  - q_proj weight: [2560, 2560]
  - k_proj weight: [640, 2560]
  - v_proj weight: [640, 2560]
  - o_proj weight: [2560, 2560]

## 变换接口

`get_transform(layer_idx, proj_type, A_calib, W, device)` 返回：
- `(T, T_inv, mean)` 其中：
  - `T [D,D]`：激活变换，`A_t = (A - mean) @ T`
  - `T_inv [D,D]`：权重变换，`W_t = W @ T_inv.T`
  - `mean [D]`：激活均值用于 bias correction（可以是 zeros）
  - 必须满足：`T @ T_inv ≈ I`
- 或 `None`（等同于不做变换）

---

## Baseline（Block Hadamard 128）

```
avg_snr_improvement_db:    -0.9283
worst_case_improvement_db: -2.8476

  L2_q_proj: base=21.40 dB  rot=21.09 dB  imp=-0.31 dB
  L2_k_proj: base=19.76 dB  rot=19.35 dB  imp=-0.41 dB
  L2_v_proj: base=16.61 dB  rot=13.76 dB  imp=-2.85 dB  ← 最大问题
  L2_o_proj: base=16.35 dB  rot=16.19 dB  imp=-0.15 dB
```

**目标**：avg_snr_improvement_db > 0（优于无变换）

---

## 实验循环

重复以下步骤：

1. **思考** — 查看 logs/results.tsv 中已有结果，决定下一步尝试什么
2. **编辑** — 修改 `optimize.py`
3. **运行**：
   ```bash
   CUDA_VISIBLE_DEVICES=1 conda run -n py311 --no-capture-output python optimize.py
   ```
4. **读取结果**：检查输出中的 `avg_snr_improvement_db / worst_case / transform_type / elapsed_seconds`
   - 无输出（崩溃）→ 检查错误，最多重试 3 次，否则放弃进入下一方向
5. **记录** → 追加到 `logs/results.tsv`（不 commit）。对于重要的结果，根据exp_id，保存到data/{exp_id}。例如训练超参数和配置，训练结果的矩阵。
6. **保留或回滚**：
   - 若 avg_snr_improvement_db 改善 → `git add optimize.py && git commit -m "<tag>: <desc>"`
   - 否则 → `git checkout -- optimize.py`

---

## logs/results.tsv 格式

```
tag	avg_snr_improvement_db	worst_db	transform_type	notes
001	-0.9283	-2.8476	block_hadamard_128	baseline
```
