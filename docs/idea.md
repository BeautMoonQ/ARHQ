# Act-Residual Hessian Low-Rank Residual Quantization

## 1. Problem Setting

考虑一个线性层：

`Y = A W^T`

其中：

- `A in R^(N x D)`：输入 activation
- `W in R^(D_out x D)`：weight
- `Y in R^(N x D_out)`：输出

如果直接对 activation 和 weight 做低比特量化，则部署形式为：

`Y_hat ~= Q(A) Q(W)^T`

此时量化误差会同时来自：

- activation 的失真
- weight 的失真
- 两者在输出空间中的耦合

为了降低这部分误差，可以先把权重拆成两部分：

`W = W_sig + W_res`

其中：

- `W_sig`：由低秩分支精确保留的信号部分
- `W_res`：交给量化主分支处理的残差部分

于是部署时改写为：

`Y_hat = Q(A) Q(W_res)^T + A W_sig^T`

本方法的核心问题是：

`如何选择最值得进入低秩分支的 rank-r 子空间`


## 2. Difficulty Transfer Form

在本项目中，low-rank 提取通常不是直接在原始 `A, W` 上进行，而是先做 difficulty transfer：

`A' = ((A - mu) Q)_mix odot s`

`W' = ((W Q)_mix) oslash s`

其中：

- `mu in R^D`：activation 的离线中心
- `Q`：分块正交变换
- `_mix`：可选的 block-pair mixing
- `s in R^D`：per-channel scale

经过这一步之后，真正用于 low-rank 分解和量化的是 `A'` 与 `W'`，于是部署式变成：

`Y_hat = Q(A') Q(W'_res)^T + A' W'_sig^T + mu W^T`

为了降低在线计算开销，通常把 low-rank 项写成双因子形式：

`W'_sig = B_r A_fac^T`

其中：

- `A_fac in R^(D x r)`
- `B_r in R^(D_out x r)`

因此最终部署形式为：

`Y_hat = Q(A') Q(W'_res)^T + (A' A_fac) B_r^T + mu W^T`

这就是完整的“量化主分支 + 低秩补偿分支”结构。


## 3. Baseline: SVDQuant-Style Low-Rank Residual

baseline 采用 `SVDQuant` 风格的 low-rank residual 思路：对变换后的权重 `W'` 做截断 SVD，直接提取其主奇异子空间。

优化目标为：

`min_rank(W'_sig)<=r ||W' - W'_sig||_F^2`

对 `W'` 做分解：

`W' = U Sigma V^T`

取前 `r` 个奇异方向，得到：

`W'_sig = U_r Sigma_r V_r^T`

`W'_res = W' - W'_sig`

其含义是：

- 选取 `W'` 中 Frobenius 能量最大的 `r` 个方向
- 这些方向交给浮点 low-rank 分支
- 剩余残差交给量化器

因此 baseline 的部署形式为：

`Y_hat = Q(A') Q(W'_res)^T + A' W'_sig^T + mu W^T`

这条线的优点是简单直接，但它只看 `W'` 自身的主能量方向，不显式考虑：

- 输入数据分布
- 输出敏感度
- activation 量化残差的结构


## 4. Motivation for Act-Residual Hessian

如果只按 `W'` 的主奇异方向选 low-rank 子空间，那么默认所有输入方向的重要性相同。

但线性层真正的输出误差由下式决定：

`A' (W' - W'_sig)^T`

因此一个更合理的 low-rank 目标应该同时回答两个问题：

1. 哪些输入方向本身更重要
2. 哪些输入方向更容易被量化器破坏

第一个问题由 activation 的二阶统计决定，第二个问题由 activation 量化残差的结构决定。`Act-Residual Hessian` 的核心，就是把这两部分信息合并到同一个度量矩阵中，再据此选择 rank-`r` 子空间。


## 5. Method Definition

### 5.1 Activation Hessian

先计算 transformed activation 的二阶统计：

`H = A'^T A' / N`

它刻画的是：

- 数据中哪些输入方向更常出现
- 哪些方向上的 weight 误差更容易传递到输出


### 5.2 Activation Quantization Residual

定义 activation 的量化残差：

`E = A' - Q(A')`

再构造其协方差：

`R = E^T E / N`

`R` 刻画的是：

- 哪些 activation 方向更容易被量化器破坏
- 哪些量化误差具有明显结构，而不是均匀噪声


### 5.3 Act-Residual Hessian

将两者融合，得到新的度量矩阵：

`H_res = H + beta R`

其中：

`beta = mean(diag(H)) / mean(diag(R))`

这个 `beta` 仅用于对齐 `H` 与 `R` 的整体尺度，避免残差项过强或过弱。

于是最终优化目标为：

`min_rank(W'_sig)<=r ||(W' - W'_sig) H_res^(1/2)||_F^2`

等价写法为：

`min_rank(W'_sig)<=r tr((W' - W'_sig) H_res (W' - W'_sig)^T)`

这说明该方法选取的 low-rank 子空间，不是简单的 weight 主子空间，而是：

- 对输出更敏感的方向
- 同时也是 activation 量化误差更集中的方向


## 6. Closed-Form Construction

给定：

`H_res = U Lambda U^T`

先构造：

`H_res^(1/2) = U Lambda^(1/2) U^T`

`H_res^(-1/2) = U Lambda^(-1/2) U^T`

然后定义加权矩阵：

`M = W' H_res^(1/2)`

对 `M` 做 rank-`r` 的 SVD：

`M ~= B_r A_tilde^T`

再映回原坐标，得到：

`A_fac = H_res^(-1/2) A_tilde`

`W'_sig = B_r A_fac^T`

`W'_res = W' - W'_sig`

因此，最终保留下来的 `r` 个方向，并不是由 `W'` 自身的奇异值直接决定，而是由 `W'` 在 `H_res` 度量下的重要性决定。

当 `r = 128` 时，表示：

- 在 `H_res` 诱导的度量空间中
- 选出最值得由低秩分支保留的 `128` 个方向


## 7. Quantized Deployment Form

完成 low-rank 分解后，部署时不再量化整个 `W'`，而只量化残差部分：

`W'_res = W' - W'_sig`

因此主分支为：

`Q(A') Q(W'_res)^T`

low-rank 补偿分支为：

`A' W'_sig^T`

将 `W'_sig = B_r A_fac^T` 代入后，可写为：

`Y_hat = Q(A') Q(W'_res)^T + (A' A_fac) B_r^T + mu W^T`

也就是说，在线计算分成三部分：

1. 对 transformed activation 做量化：`Q(A')`
2. 对 transformed residual weight 做量化：`Q(W'_res)`
3. 用浮点 low-rank 分支补回最关键的 rank-`r` 子空间

这里量化函数 `Q()` 始终只作用在主分支残差上，而不会作用在已经被 low-rank 分支吸收的 `W'_sig` 上。


## 8. Interpretation

`Act-Residual Hessian` 的本质不是“再做一次 SVD”，而是：

- 先用 `H_res` 重新定义什么叫重要方向
- 再在这个度量下选取 low-rank 子空间

因此它和 `SVDQuant` baseline 的本质区别在于：

- `SVDQuant` baseline：保留 `W'` 自身主能量方向
- `Act-Residual Hessian`：保留“输出敏感且易被量化器破坏”的方向

可以把两者概括为：

- baseline 更偏 `weight-centric`
- `Act-Residual Hessian` 更偏 `output-aware + quantization-aware`


## 9. Why It Can Beat the SVDQuant Baseline

如果量化误差只是平滑、均匀、近似各向同性的噪声，那么 `R = E^T E / N` 提供的新增信息有限，此时简单的 `SVDQuant` baseline 往往已经很强。

但如果量化误差具有显著结构，例如：

- block shared scale
- 非均匀 codebook
- outlier 诱导的分辨率塌缩
- zero occupancy 或 saturation 结构

那么 `E = A' - Q(A')` 往往会在某些方向上集中，而不是近似均匀散开。

此时：

- `R` 不再只是噪声统计
- 而是在告诉 low-rank 分支“哪些关键方向最容易被量化器破坏”

这正是 `Act-Residual Hessian` 相比 `SVDQuant` baseline 可能更强的根本原因。


## 10. Summary

`Act-Residual Hessian Low-Rank Residual Quantization` 的完整流程可以概括为：

1. 对原始线性层做 difficulty transfer，得到 `A'` 和 `W'`
2. 计算 activation 二阶统计 `H = A'^T A' / N`
3. 计算 activation 量化残差 `E = A' - Q(A')`
4. 构造残差协方差 `R = E^T E / N`
5. 融合得到 `H_res = H + beta R`
6. 在 `H_res` 度量下，对 `W'` 提取 rank-`r` low-rank 信号部分 `W'_sig`
7. 将剩余部分写成 `W'_res = W' - W'_sig`
8. 部署时执行：

   `Y_hat = Q(A') Q(W'_res)^T + (A' A_fac) B_r^T + mu W^T`

与 `SVDQuant` baseline 相比，本方法并不满足于提取权重的主奇异方向，而是试图提取：

- 对输出最重要
- 且最容易被量化器破坏

的那部分结构。
