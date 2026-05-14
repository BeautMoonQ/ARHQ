# ARHQ 技术报告草稿

## Overview

ARHQ 指 **Activation Residual Hessian Quantization**。这里的 `Activation Residual` 不是普通激活值本身，而是激活值经过目标量化器后的量化残差：

```text
E_x = X - Q_x(X)
```

ARHQ 的核心做法是把线性层权重分离成一个参与量化主路径的 residual weight 和一个不量化的 LoRA-like 低秩分支：

```text
W = W_res + L
L = B A^T
```

推理时，`W_res` 继续与量化后的激活值一起计算，而 `L` 通过额外的低秩分支以较高精度计算：

```text
\hat{Y} = Q_x(X) Q_w(W_res)^T + X L^T
```

关键问题是如何选择这个低秩分支 `L`。ARHQ 使用 activation quantization residual 构造输入侧二次度量：

```text
R_x = E_x^T E_x / N
```

因此，`R_x` 可以看作 activation quantization residual 诱导出的 Hessian。它不同于常见的 activation Hessian：

```text
H_x = X^T X / N
```

`H_x` 衡量原始激活能量，通常对应输出重建误差；而 `R_x` 衡量激活值量化误差的能量和相关性，对应激活量化误差经过 residual weight 后的输出传播误差。

ARHQ 在这个 Activation Residual Hessian 度量下选择 LoRA 分支，使得剩余权重 `W_res = W - L` 对激活量化残差尽可能不敏感。直观上，ARHQ 把最容易放大 activation residual 的权重方向分离到不量化的低秩分支中，从而让 activation residual 对最终 linear 输出误差的影响尽可能小。

## Motivation

考虑一个线性层：

```text
Y = X W^T
```

其中 `X in R^{N x D_in}` 是输入激活，`W in R^{D_out x D_in}` 是权重。量化推理中，激活值和权重都会被量化。令：

```text
Q_x(X) = X + Delta_x
Q_w(W) = W + Delta_w
```

其中 `Delta_x` 是激活值量化误差，`Delta_w` 是权重量化误差。直接量化后的输出为：

```text
\hat{Y} = Q_x(X) Q_w(W)^T
```

因此输出误差可以展开为：

```text
\hat{Y} - Y
= (X + Delta_x)(W + Delta_w)^T - X W^T
= Delta_x W^T + X Delta_w^T + Delta_x Delta_w^T
```

这个公式说明量化误差主要来自三部分：

```text
Delta_x W^T          : 激活值量化误差经过权重传播
X Delta_w^T          : 权重量化误差经过激活传播
Delta_x Delta_w^T    : 激活值和权重量化误差的交叉项
```

ARHQ 的出发点是：在低秩分支开销可接受的前提下，把权重分离为一个不量化的低秩分支和一个参与主路径量化的 residual weight：

```text
W = W_res + L
L = B A^T, rank(L) <= r
```

推理形式变为：

```text
\hat{Y} = Q_x(X) Q_w(W_res)^T + X L^T
```

其中 `L` 作为 LoRA-like 低秩分支以较高精度计算，`W_res` 继续走量化主路径。

本文先考虑一个简化问题：忽略 `W_res` 的权重量化误差，只关注激活值量化误差如何通过 residual weight 传播。此时近似推理为：

```text
\hat{Y} = Q_x(X) W_res^T + X L^T
```

代入 `W_res = W - L`：

```text
\hat{Y}
= Q_x(X)(W - L)^T + X L^T
```

令：

```text
E_x = X - Q_x(X)
```

则：

```text
Y - \hat{Y}
= X W^T - Q_x(X)(W - L)^T - X L^T
= (X - Q_x(X))(W - L)^T
= E_x (W - L)^T
```

因此，ARHQ 的核心目标是选择一个低秩矩阵 `L`，使激活值量化误差经过 residual weight 后的输出误差尽可能小：

```text
min_{rank(L) <= r} || E_x (W - L)^T ||_F^2
```

这个目标体现了两个动机：

1. 利用 `L` 和 `W_res` 的分离，降低激活值量化误差 `E_x` 在输出端的传播。
2. 先忽略 `W_res` 的权重量化误差，把问题简化为一个有闭式解的 weighted low-rank approximation。

需要强调的是，优化对象不是 `E_x W^T`。因为 `X`、`Q_x(X)` 和 `W` 在校准阶段都是固定的，`E_x W^T` 与低秩分支 `L` 无关，无法通过选择 `L` 被优化。真正可控的是 residual weight：

```text
W_res = W - L
```

因此目标必须写成：

```text
E_x (W - L)^T
```

## ARHQ Objective

定义 activation quantization residual covariance：

```text
G_x = E_x^T E_x / N
```

则目标可以改写为：

```text
|| E_x (W - L)^T ||_F^2
= Tr((W - L) E_x^T E_x (W - L)^T)
= N · Tr((W - L) G_x (W - L)^T)
```

忽略常数 `N`，得到：

```text
min_{rank(L) <= r} || (W - L) G_x^{1/2} ||_F^2
```

这里的 `G_x` 可以看作由激活值量化残差诱导出的 residual Hessian。它不是普通 activation Hessian：

```text
H_x = X^T X / N
```

而是：

```text
G_x = (X - Q_x(X))^T (X - Q_x(X)) / N
```

因此，ARHQ 不是在重建 `W` 本身，而是在量化残差最敏感的输入方向上重建 `W`。如果某些输入通道的激活值量化误差更大，或者这些误差之间有更强相关性，它们会在 `G_x` 中获得更高权重。

这也是 ARHQ 与普通 SVDQuant 的关键区别。SVDQuant 的低秩分解通常求解：

```text
min_{rank(L) <= r} || W - L ||_F^2
```

而 ARHQ 求解的是：

```text
min_{rank(L) <= r} || (W - L) G_x^{1/2} ||_F^2
```

也就是说，ARHQ 的低秩分支不是优先重建权重能量最大的方向，而是优先消除激活量化误差最容易造成输出偏差的方向。

## Weighted Decomposition

对 `G_x` 做特征分解：

```text
G_x = U diag(lambda) U^T
```

其中 `U` 是正交矩阵。为保证数值稳定，对特征值加入很小的正则项：

```text
lambda_i <- clamp(lambda_i, eps)
```

构造：

```text
G_sqrt     = U diag(sqrt(lambda)) U^T
G_inv_sqrt = U diag(1 / sqrt(lambda)) U^T
```

二者满足：

```text
G_sqrt G_sqrt = G_x
G_sqrt G_inv_sqrt ~= I
```

注意不是：

```text
G_sqrt G_inv_sqrt = G_x
```

而是 `G_sqrt` 自身平方后得到 `G_x`。

将优化目标写为：

```text
min_{rank(L) <= r} || W G_sqrt - L G_sqrt ||_F^2
```

令：

```text
M = W G_sqrt
```

对 `M` 做 rank-r SVD：

```text
M ~= B C^T
```

其中：

```text
B in R^{D_out x r}
C in R^{D_in x r}
```

为了回到原始权重空间，需要：

```text
L G_sqrt = B C^T
```

因此：

```text
L = B C^T G_inv_sqrt
```

写成 LoRA-like 形式：

```text
L = B A^T
A = G_inv_sqrt C
```

最终得到：

```text
B_r   = B
A_fac = A
L     = B_r A_fac^T
W_res = W - L
```

推理时使用：

```text
\hat{Y} = Q_x(X) Q_w(W_res)^T + (X A_fac) B_r^T
```

其中：

```text
Q_x(X) Q_w(W_res)^T
```

是量化主分支，

```text
(X A_fac) B_r^T
```

是不量化的低秩分支。

## Calibration and Decomposition Pipeline

ARHQ 的离线流程如下。

首先收集校准激活。对每个目标 linear layer，使用校准数据跑 prefill，并保存该 linear 的输入 activation：

```text
X_calib in R^{N_calib x D_in}
```

同时保存当前模型中的原始权重：

```text
W in R^{D_out x D_in}
```

然后使用目标量化器对校准激活做 fake quant：

```text
X_q = Q_x(X_calib)
```

计算激活值量化残差：

```text
E_x = X_calib - X_q
```

计算 residual Hessian：

```text
G_x = E_x^T E_x / N_calib
```

然后构造：

```text
G_sqrt
G_inv_sqrt
M = W G_sqrt
```

对 `M` 做截断 SVD：

```text
M ~= B C^T
```

恢复 LoRA 因子：

```text
B_r   = B
A_fac = G_inv_sqrt C
```

构造分离后的 residual weight：

```text
W_res = W - B_r A_fac^T
```

最终保存推理所需的中间结果：

```text
B_r
A_fac
W_res
```

如果后续推理阶段选择从当前模型权重重新计算 residual，则也可以只保存：

```text
B_r
A_fac
```

并在加载模型后计算：

```text
W_res = W_current - B_r A_fac^T
```

这可以避免保存权重和推理时加载权重之间存在 dtype 或 checkpoint 差异时造成不一致。

## Smoothing Extension

ARHQ 直接优化激活值量化误差的传播，但它并不会让 `E_x` 本身变小。如果激活值量化误差很大，即使选择了较好的 `W_res`，最终误差仍然可能被 `E_x` 的大小限制。

因此引入 smoothing。给定 per-channel scale `s in R^{D_in}`，做等价变换：

```text
X_s = X / s
W_s = W * s
```

因为：

```text
X_s W_s^T = X W^T
```

所以 smoothing 不改变原始浮点线性层的输出，但会改变 activation 和 weight 的量化难度。

在 smoothing 之后，ARHQ 在变换后的空间中执行：

```text
E_s = X_s - Q_x(X_s)
G_s = E_s^T E_s / N
```

然后分解：

```text
M_s = W_s G_s^{1/2}
M_s ~= B C^T
A_s = G_s^{-1/2} C
L_s = B A_s^T
W_res_s = W_s - L_s
```

推理时：

```text
\hat{Y}
= Q_x(X / s) Q_w(W_res_s)^T + ((X / s) A_s) B^T
```

其中 `W_res_s` 是 smoothing 后权重空间中的 residual weight。

ARHQ 与 smoothing 是互补的：

1. smoothing 试图减小或重新分配激活值量化误差本身。
2. ARHQ 试图降低剩余激活值量化误差在输出端的传播。

## Quantization Adaptivity

ARHQ 的 residual Hessian 直接由真实量化器的误差构造：

```text
G_x = (X - Q_x(X))^T (X - Q_x(X)) / N
```

因此它天然依赖具体量化方式。对于 block-wise 量化，不同 block 的 scale 会导致不同通道和不同 token 上的误差分布不同；对于 NVFP4 这类非均匀量化，误差也不再接近简单的均匀噪声。

ARHQ 不需要显式假设量化误差服从某种解析分布，而是直接从校准数据和目标量化器中测量 `E_x`。因此，当量化器从 uniform int4 变成 block-wise int4，或者变成 NVFP4 这类带非均匀 codebook 和 block scale 的格式时，`G_x` 会随之变化，分解出的低秩分支也会适配新的误差结构。

这也是 ARHQ 相比普通 SVDQuant 的优势之一。SVDQuant 只看 `W` 的低秩结构；ARHQ 看的是：

```text
W 在 activation quantization residual 度量下的有效结构
```

因此它更适合用来处理实际量化器带来的结构化误差。

## Relation to SVDQuant

SVDQuant 可以看作不使用校准激活、不使用量化误差信息的低秩权重分解：

```text
W ~= L
W_res = W - L
```

它的目标是尽可能保留 `W` 自身的大能量方向。这个目标在权重量化误差主导时可能有效，但它并不区分哪些输入方向上的误差会被 activation quantization 放大。

ARHQ 则从输出误差公式出发，把 activation quantization residual 显式放进目标函数。它不要求 `L` 是 `W` 的最佳 Frobenius 重建，而要求 `W_res = W - L` 对激活值量化误差不敏感。

换句话说：

```text
SVDQuant:  低秩分支重建 W
ARHQ:      低秩分支消除 E_x 敏感方向上的 W
```

这解释了为什么在不使用 smoothing 时，ARHQ 也能明显优于 raw SVDQuant。

## SNR Results

以下 SNR 结果来自 Qwen3-4B-Thinking-2507 的 36 层 attention projection，rank=128。表中的 ARHQ raw / ARHQ smoothing 对应实验脚本中的 `r_only raw` / `r_only smoothing`。SNR 越高越好。

### Method SNR

| Scope | ARHQ Raw | ARHQ Smoothing | SVDQuant Raw | SVDQuant Smoothing |
|---|---:|---:|---:|---:|
| q/k/v/o average | 24.4229 | 24.7269 | 22.6314 | 24.2742 |
| q_proj | 28.3389 | 28.6415 | 25.7253 | 28.1420 |
| k_proj | 28.3682 | 28.6888 | 25.8594 | 27.9529 |
| v_proj | 21.4883 | 21.7858 | 19.6953 | 21.4014 |
| o_proj | 19.4964 | 19.7917 | 19.2456 | 19.6006 |

### Improvement over Baseline

| Scope | ARHQ Raw | ARHQ Smoothing | SVDQuant Raw | SVDQuant Smoothing |
|---|---:|---:|---:|---:|
| q/k/v/o average | +4.0828 | +3.7906 | +2.5627 | +3.3378 |
| q_proj | +5.1668 | +4.7555 | +3.2661 | +4.2560 |
| k_proj | +6.5745 | +6.1158 | +4.2142 | +5.3799 |
| v_proj | +2.9964 | +2.6233 | +1.5308 | +2.2389 |
| o_proj | +1.5936 | +1.6675 | +1.2397 | +1.4764 |

这些结果支持两个观察：

1. 不使用 smoothing 时，ARHQ raw 已经可以取得较高 SNR，并且相比 SVDQuant raw 有明显优势。q/k/v/o 平均上，ARHQ raw 为 `24.4229 dB`，SVDQuant raw 为 `22.6314 dB`。
2. smoothing 进一步提升 ARHQ。q/k/v/o 平均上，ARHQ smoothing 为 `24.7269 dB`，高于 ARHQ raw 的 `24.4229 dB`，也高于 SVDQuant smoothing 的 `24.2742 dB`。

## Small-Scale ZebraLogic Evaluation

除离线 SNR 外，还在 ZebraLogic 上做了小规模生成式评测。实验设置如下：

```text
Model: Qwen3-4B-Thinking-2507
Dataset: ZebraLogic
Seed: 0
Temperature: 0.6
Top-p: 0.95
Top-k: 20
Number of evaluated puzzles: 140
Repeat: 0
```

当前结果如下：

| Method | Puzzle Acc | Cell Acc | Cell Recall |
|---|---:|---:|---:|
| bf16 | 130/140 = 92.86% | 2671/2732 = 97.77% | 2671/2837 = 94.15% |
| nvfp4 | 105/140 = 75.00% | 2279/2625 = 86.82% | 2279/2783 = 81.89% |
| awq4bit | 125/140 = 89.29% | 2589/2771 = 93.43% | 2589/2873 = 90.11% |
| svdquant_r128 | 127/140 = 90.71% | 2643/2766 = 95.55% | 2643/2873 = 91.99% |
| arhq_r128_all | 130/140 = 92.86% | 2660/2779 = 95.72% | 2660/2873 = 92.59% |

在这 140 个问题上，`arhq_r128_all` 的 puzzle accuracy 为 `130/140 = 92.86%`，高于 `awq4bit` 的 `125/140 = 89.29%` 和 `svdquant_r128` 的 `127/140 = 90.71%`，并与 bf16/vLLM 的 `130/140 = 92.86%` 持平。不过该实验仍处于早期阶段，样本数只有 140 个问题，结果可能存在噪声。因此这组结果只能作为初步信号，不能作为最终结论。

更稳健的验证需要：

1. 扩展到完整 ZebraLogic 测试集。
2. 固定多组随机种子并报告均值和方差。
3. 增加 MATH-500 等更多任务。
4. 对 attention-only、ffn-only、all-layer 设置分别评测。

## Current Limitations

当前 ARHQ 推导为了得到闭式分解，显式忽略了 residual weight 的量化误差。真实推理为：

```text
\hat{Y} = Q_x(X) Q_w(W_res)^T + X L^T
```

而不是：

```text
\hat{Y} = Q_x(X) W_res^T + X L^T
```

令：

```text
W_res = W - L
Q_w(W_res) = W_res + E_res
Q_x(X) = X + E_q
```

真实误差中仍然包含：

```text
X E_res^T
E_q E_res^T
```

因此 ARHQ 可能出现一个 trade-off：它降低了激活值量化误差经过 `W_res` 的传播，但可能让 `W_res` 本身更难量化。当前方法没有直接把这一项纳入闭式目标。

后续方向可以考虑 residual-weight quantization-aware 的扩展，例如：

```text
min_L ||E_x (W - L)^T||_F^2
    + lambda ||X ((W - L) - Q_w(W - L))^T||_F^2
```

但该目标包含 `Q_w(W - L)`，量化算子依赖 `L`，不再有简单闭式解。可能的实现包括固定点迭代、STE 训练 LoRA 因子，或只在分解后做轻量校正。

## Summary

ARHQ 从量化线性层的输出误差出发，针对 activation quantization residual 构造 residual Hessian，并在该度量下分离出不量化的低秩 LoRA 分支和参与量化主路径的 `W_res`。相比 SVDQuant，ARHQ 的低秩分支不是为了重建 `W` 本身，而是为了降低激活值量化误差在输出端的传播。

在当前 NVFP4 SNR 结果中，ARHQ raw 已经明显优于 SVDQuant raw；加入 smoothing 后，ARHQ 进一步提升，并超过 SVDQuant smoothing。小规模 ZebraLogic 结果也显示 `arhq_r128_all` 在 140 个问题上取得了较好的 puzzle accuracy，但仍需要更大规模和多 seed 的验证。
