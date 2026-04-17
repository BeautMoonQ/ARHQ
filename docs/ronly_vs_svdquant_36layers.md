# 全 36 层 Sweep: smooth + R_only vs smooth + SVDQuant

## 实验配置

- **模型**: Qwen3-4B-Thinking-2507 (36 layers)
- **投影**: q_proj [4096,2560], k_proj [1024,2560], v_proj [1024,2560], o_proj [2560,4096]
- **数据**: MATH-500 校准集, 50000 tokens/layer, 最后 2048 tokens 为评估集
- **量化**: NVFP4 E2M1, per-block(16) FP8 E4M3 scale (1D block)
- **指标**: 绝对 SNR (dB) = 20·log10(||Y_true|| / ||Y_hat - Y_true||), 越高越好
- **设备**: 2×RTX 6000 Ada (GPU2: layers 0-17, GPU3: layers 18-35), 共 ~12 分钟

## 方法说明

| 方法 | 说明 |
|------|------|
| **smooth + SVDQuant** | SmoothQuant scaling → 截断 SVD (≈ SVDQuant 论文配置) |
| **smooth + R_only** | SmoothQuant scaling → R-weighted SVD, 其中 R = E^T E / N, E = A - Q(A) |

**R_only** 是 ARHQ 方法的简化版本：去掉 activation Hessian H，仅用量化残差协方差 R 做加权。
由之前 9 层消融实验得出 R_only 是最优配置（H 反而有害），本次在全部 36 层验证。

**部署公式**:
```
Y ≈ Q(A') @ Q(W_res)^T + (A' @ A_fac) @ B_r^T
```
其中 `A' = A / scale`, `W_res = W·scale - B_r @ A_fac^T`。低秩分支 (A' @ A_fac) @ B_r^T 在 FP16 精度计算。

---

## 1. 按 Rank 平均结果 (144 个 layer×proj 组合 per rank)

| rank | SVDQuant (dB) | R_only (dB) | Δ (dB) |
|------|:---:|:---:|:---:|
| 32 | 22.43 | 22.84 | **+0.41** |
| 64 | 23.10 | 23.60 | **+0.50** |
| 128 | 24.06 | 24.64 | **+0.58** |
| 256 | 25.62 | 26.27 | **+0.65** |

**趋势**: rank 越大，R_only 优势越明显 (+0.41 → +0.65 dB)。

---

## 2. 按投影类型分解 (rank=128, 36 层平均)

| proj | SVDQuant (dB) | R_only (dB) | Δ (dB) |
|------|:---:|:---:|:---:|
| q_proj | 27.50 | 28.18 | **+0.68** |
| k_proj | 28.02 | 28.89 | **+0.88** |
| v_proj | 21.01 | 21.51 | **+0.50** |
| o_proj | 19.73 | 19.98 | **+0.25** |

**观察**: k_proj 获益最大 (+0.88 dB), o_proj 最小 (+0.25 dB)。v_proj 和 o_proj 的 SNR 绝对值较低，说明这两个投影的量化误差本身更大。

---

## 3. 胜率统计

| rank | R_only 胜出 | 总数 | 胜率 |
|------|:---:|:---:|:---:|
| 32 | 143 | 144 | **99.3%** |
| 64 | 143 | 144 | **99.3%** |
| 128 | 142 | 144 | **98.6%** |
| 256 | 142 | 144 | **98.6%** |

### SVDQuant 胜出的少数案例

仅 6 个 case，全部集中在 o_proj 且差距极小：

| rank | 位置 | SVDQuant 领先 |
|------|------|:---:|
| 32 | L19/o_proj | -0.02 dB |
| 64 | L3/o_proj | -0.04 dB |
| 128 | L1/o_proj | -0.01 dB |
| 128 | L3/o_proj | -0.05 dB |
| 256 | L1/o_proj | -0.01 dB |
| 256 | L3/o_proj | -0.12 dB |

---

## 4. 逐层对比 (rank=128, 4 proj 平均)

| Layer | Δ (dB) | | Layer | Δ (dB) | | Layer | Δ (dB) |
|:---:|:---:|---|:---:|:---:|---|:---:|:---:|
| 0 | **+1.48** | | 12 | **+0.89** | | 24 | +0.58 |
| 1 | +0.40 | | 13 | +0.46 | | 25 | +0.55 |
| 2 | +0.50 | | 14 | +0.54 | | 26 | +0.36 |
| 3 | +0.72 | | 15 | +0.47 | | 27 | +0.48 |
| 4 | +0.32 | | 16 | +0.42 | | 28 | **+0.88** |
| 5 | +0.65 | | 17 | +0.55 | | 29 | +0.70 |
| 6 | +0.51 | | 18 | +0.34 | | 30 | +0.73 |
| 7 | +0.48 | | 19 | +0.41 | | 31 | **+1.14** |
| 8 | +0.42 | | 20 | +0.39 | | 32 | **+0.99** |
| 9 | +0.46 | | 21 | +0.33 | | 33 | **+0.93** |
| 10 | +0.45 | | 22 | +0.34 | | 34 | +0.56 |
| 11 | +0.44 | | 23 | +0.36 | | 35 | +0.49 |

**观察**: 首尾层优势更显著 (Layer 0: +1.48, Layer 31: +1.14)，中间层相对稳定 (+0.3~0.5 dB)。

---

## 5. 关键结论

1. **R_only 全面优于 SVDQuant**: 在全部 36 层、4 种投影、4 种 rank 的 576 组对比中，**99% 的情况下 R_only 更优**。
2. **优势随 rank 增大**: +0.41 dB (rank=32) → +0.65 dB (rank=256)，说明 R-weighted SVD 选出的低秩子空间质量更高。
3. **少数失败 case 差距极小**: SVDQuant 仅在 6 个 o_proj case 中微弱胜出 (最大 -0.12 dB)，不影响整体结论。
4. **方法简洁**: R_only 不需要 H 矩阵，不需要 β 超参数，唯一额外计算是量化残差协方差 R = E^T E / N。

---

## 6. 保存的数据

### 目录结构
```
results/
├── summary/
│   ├── ronly_vs_svdq_all_layers.csv       # 合并后的完整 CSV (1152 行)
│   ├── ronly_vs_svdq_all_layers_gpu2.csv  # GPU2 原始结果 (layers 0-17)
│   └── ronly_vs_svdq_all_layers_gpu3.csv  # GPU3 原始结果 (layers 18-35)
├── layer_results/
│   ├── layer_0/
│   │   ├── q_proj_svdquant_smoothing_rank32.pt
│   │   ├── q_proj_svdquant_smoothing_rank64.pt
│   │   ├── q_proj_svdquant_smoothing_rank128.pt
│   │   ├── q_proj_svdquant_smoothing_rank256.pt
│   │   ├── q_proj_r_only_smoothing_rank32.pt
│   │   ├── ...  (共 32 个 .pt 文件 per layer: 4 proj × 4 rank × 2 method)
│   ├── layer_1/
│   │   └── ...
│   └── layer_35/
│       └── ...
```

**共 1152 个 .pt 文件** (36 层 × 32 files/层)

### 每个 .pt 文件包含

| 字段 | 形状 | 说明 |
|------|------|------|
| `B_r` | [D_out, rank] (fp16) | 低秩左因子 |
| `A_fac` | [D_in, rank] (fp16) | 低秩右因子 (activation 侧) |
| `W_res` | [D_out, D_in] (fp16) | 残差权重 (部署时用 nvfp4 量化) |
| `scale` | [D_in] (fp16) | SmoothQuant channel scale |
| `alpha` | float | SmoothQuant 最优 α |
| `rank` | int | 低秩维度 |
| `method` | str | "svdquant" 或 "r_only" |
| `layer` | int | 层索引 |
| `proj` | str | 投影类型 |
| `snr_baseline_db` | float | 仅量化 (无低秩) 的 SNR |
| `snr_method_db` | float | 量化+低秩 的 SNR |

### 使用方式 (模拟 nvfp4 + lora 推理)

```python
data = torch.load("results/layer_results/layer_0/q_proj_r_only_smoothing_rank128.pt")
scale = data["scale"].float().cuda()
B_r = data["B_r"].float().cuda()
A_fac = data["A_fac"].float().cuda()
W_res = data["W_res"].float().cuda()

# 推理
A_smooth = A / scale                              # SmoothQuant scaling
Y_main = nvfp4_quantize(A_smooth) @ nvfp4_quantize(W_res).T   # 量化主路径
Y_lr = (A_smooth @ A_fac) @ B_r.T                # 低秩补偿 (FP16)
Y = Y_main + Y_lr
```
