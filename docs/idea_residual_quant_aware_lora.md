# Residual Quant-Aware LoRA 分解想法

本文记录一个后续实验想法：在当前 R-only / ARHQ 闭式分解基础上，把 residual weight 的量化误差也纳入低秩分解目标。

## 当前 R-only 目标

设：

```text
X       : 校准激活，shape [N, D_in]
X_q     : Q(X)，activation fake quant 后 dequant 的结果
E_x     : X - X_q，activation 量化残差
G_act   : E_x.T @ E_x / N，activation 量化残差协方差

W       : 原始权重，shape [D_out, D_in]
L       : rank-r 低秩矩阵，L = B @ A.T
W_res   : W - L
```

当前 R-only 选择 `L` 的目标可以写成：

```text
min_{rank(L) <= r} ||E_x @ (W - L).T||_F^2
```

等价于：

```text
min_{rank(L) <= r} ||(W - L) @ G_act^{1/2}||_F^2
```

其中：

```text
G_act = E_x.T @ E_x / N
```

直观理解：`E_x` 和 `W` 都固定，只有 `L` 可变。R-only 会选择一个最好的 rank-r 低秩矩阵 `L`，让剩余权重 `W_res = W - L` 在 activation 量化残差敏感方向上的输出误差尽量小。

如果目标写成 `||E_x @ W.T||`，它和 `L` 无关，无法优化；所以目标必须使用 residual：

```text
W - L
```

## 为什么还要考虑 residual weight 量化

实际推理不是：

```text
Q(X) @ W_res.T + X @ L.T
```

而是：

```text
Q(X) @ Q(W_res).T + X @ L.T
```

其中：

```text
W_res = W - L
```

因此 residual weight 本身也会被量化。令：

```text
E_w = W_res - Q(W_res)
```

忽略 activation 和 weight 量化误差的二阶交叉项时，输出误差近似包含两部分：

```text
E_x @ W_res.T
X @ E_w.T
```

当前 R-only 只显式考虑了第一项：

```text
E_x @ (W - L).T
```

更完整的目标应考虑第二项：

```text
X @ ((W - L) - Q(W - L)).T
```

可写成：

```text
min_L ||E_x @ (W - L).T||_F^2
    + λ ||X @ ((W - L) - Q(W - L)).T||_F^2
```

难点是第二项包含：

```text
Q(W - L)
```

量化算子不可导、分段常数，而且依赖 `L`，所以不再有简单的一次 weighted SVD 闭式解。

## 方案一：固定点迭代

核心思路：每一轮根据当前低秩 `L_t` 计算 residual weight 的量化误差，再把这个误差信息加入下一轮 weighted SVD 的度量矩阵。

初始化：

```text
L_0 = R-only 或 SVDQuant 的闭式分解结果
```

固定不变的 activation residual 度量：

```text
E_x = X - Q(X)
G_act = E_x.T @ E_x / N
```

注意：`G_act` 不随迭代变化，因为它只由校准激活和 activation quantizer 决定，和 `L_t` 无关。

第 `t` 轮：

```text
W_res_t = W - L_t
W_res_q = Q(W_res_t)
E_w_t   = W_res_t - W_res_q
G_w_t   = E_w_t.T @ E_w_t / D_out
```

组合新的输入维度度量：

```text
G_total_t = G_act + beta_w_t * G_w_t
```

`beta_w_t` 可每轮重算，用于对齐量纲，例如：

```text
beta_w_t = mean(diag(G_act)) / mean(diag(G_w_t))
```

也可以第一版先固定为：

```text
beta_w_t = 1
```

然后做一次 weighted SVD：

```text
G_sqrt_t     = G_total_t^{1/2}
G_inv_sqrt_t = G_total_t^{-1/2}
M_t          = W @ G_sqrt_t
M_t          ~= B_t @ C_t.T
A_t          = G_inv_sqrt_t @ C_t
L_{t+1}      = B_t @ A_t.T
```

伪代码：

```python
L = init_from_ronly_or_svdquant(W, X, rank)
G_act = cov(X - Q(X))

for t in range(num_iters):
    W_res = W - L
    E_w = W_res - Q(W_res)
    G_w = E_w.T @ E_w / W.shape[0]

    beta_w = mean(diag(G_act)) / mean(diag(G_w))
    G_total = G_act + beta_w * G_w

    B, A = weighted_svd(W, G_total, rank)
    L = B @ A.T
```

建议第一版实验：

```text
num_iters = 2~5
rank = 128
先只在部分 layer/proj 上验证 SNR
比较 raw R-only、smoothing R-only、固定点迭代 R-only
```

优点：

- 仍然是每轮闭式 weighted SVD。
- 不需要 STE。
- 不需要学习率等训练超参。

缺点：

- 只是近似考虑 `Q(W - L)`。
- 每轮都需要一次 `D_in x D_in` eig 和一次大矩阵 SVD，成本不低。
- 不保证单调优化真实推理 MSE。

## 方案二：直接训练低秩因子

另一条路线是直接优化和推理完全一致的目标。

令：

```text
L = B @ A.T
W_res = W - B @ A.T
```

目标：

```text
min_{A,B} ||XW.T - (Q(X) Q(W - B A.T).T + X A B.T)||_F^2
```

这就是直接训练 LoRA 因子 `A/B`，使最终 fake-quant 推理输出逼近原始输出。

可以用当前 R-only 或 SVDQuant 的闭式结果初始化：

```text
A = A_fac_from_ronly
B = B_r_from_ronly
```

然后少量 step finetune。

STE fake quant：

```python
def fake_quant_ste(x):
    x_q = nvfp4_quantize(x)
    return x + (x_q - x).detach()
```

训练伪代码：

```python
A_fac = torch.nn.Parameter(A_init.clone())
B_r = torch.nn.Parameter(B_init.clone())

for step in range(num_steps):
    x = sample_calib_batch()
    y_true = x @ W.T

    W_res = W - B_r @ A_fac.T
    W_res_q = fake_quant_ste(W_res)
    x_q = fake_quant_ste(x)

    y_hat = x_q @ W_res_q.T + (x @ A_fac) @ B_r.T
    loss = mse(y_hat, y_true)

    loss.backward()
    optimizer.step()
```

建议第一版实验：

```text
初始化：r_only smoothing
只优化 A_fac / B_r
W 固定
scale 固定
rank = 128
batch tokens = 1024 或 2048
steps = 100~500
lr = 1e-4 到 5e-4
loss = output MSE
```

优点：

- 目标和实际推理路径最一致。
- 能直接处理 `Q(W - L)` 依赖 `L` 的问题。

缺点：

- 需要 STE 或其他近似梯度。
- 训练稳定性需要调参。
- FFN 大矩阵上每步计算 `Q(W_res)` 和大矩阵输出误差，开销较高。
- 可能过拟合校准 token，需要 early stop 或 held-out eval tokens。

## 当前判断

如果只想快速验证 residual weight quantization 是否值得加入，优先尝试固定点迭代：

```text
R-only / SVDQuant 初始化
迭代 2~5 轮
看 offline SNR 和少量 ZebraLogic eval
```

如果固定点迭代有效，再考虑直接训练低秩因子，对齐真实推理目标。

## 旁路想法：不增加低秩分支时优化 activation error

另一个讨论方向是：不增加类似 `X @ L.T` 的 low-rank 分支，只沿着 activation 量化误差本身去优化。

如果没有 LoRA 分支，只考虑 activation 量化：

```text
Y = X @ W.T
Y_hat = Q(X) @ W.T
```

令：

```text
E_a = X - Q(X)
Q(X) = X - E_a
```

则：

```text
Y_hat = (X - E_a) @ W.T
      = Y - E_a @ W.T
```

所以输出误差是：

```text
Y - Y_hat = E_a @ W.T
```

一个自然目标是：

```text
min ||E_a @ W.T||_F^2
```

但如果 `X`、`W` 和 quantizer `Q` 都固定，那么 `E_a` 也是固定的，这个目标没有可优化变量。要让它成为优化问题，必须引入某种不增加额外推理分支的可调变量。

### 可调 quantizer 参数

令 quantizer 带参数：

```text
Q = Q(.; theta)
E_a(theta) = X - Q(X; theta)
```

目标可以写成：

```text
min_theta ||(X - Q(X; theta)) @ W.T||_F^2
```

`theta` 可以是：

- activation scale
- clipping threshold
- block/group 切分方式
- per-channel 或 per-block 参数

这条路线不增加计算分支，只改变 activation fake quant 的策略。

### Smoothing / scale 变换

引入 per-channel scale：

```text
X_s = X / s
W_s = W * s
```

全精度下仍保持：

```text
X_s @ W_s.T = X @ W.T
```

推理误差变成：

```text
Y - Y_hat = (X_s - Q(X_s)) @ W_s.T
```

对应目标：

```text
min_s ||(X / s - Q(X / s)) @ (W * s).T||_F^2
```

当前 `search_best_alpha` 已经是这个方向的一个受限版本：它不直接优化每个 channel 的 `s`，而是通过 SmoothQuant 公式：

```text
s = act_max^alpha / w_max^(1-alpha)
```

只搜索一维参数 `alpha`。

后续可考虑直接优化 `s`，或者做 block-aware scale，但这会引入额外约束：scale 需要能在模型结构中正确融合，不能破坏前后层语义。

### 旋转 / Hadamard 变换

更一般地，可以引入可逆变换：

```text
X_t = X @ T
W_t = W @ T^{-T}
```

保持：

```text
X_t @ W_t.T = X @ W.T
```

目标：

```text
min_T ||(X @ T - Q(X @ T)) @ (W @ T^{-T}).T||_F^2
```

其中 smoothing 是 `T` 为对角矩阵的特例。Hadamard / rotation 类方法可能改善 activation outlier 和 block 内动态范围，但需要考虑推理时是否可融合、是否会引入额外开销。

### 当前状态

这条“不要 low-rank 分支、只减小 `E_a @ W.T`”的路线目前还没有形成满意方案。

主要问题：

- 如果不改变 quantizer 或等价变换，`E_a @ W.T` 没有可优化变量。
- 如果优化 scale / clipping / grouping，容易变成 quantizer search，需要和硬件格式约束绑定。
- 如果引入旋转或 dense 变换，可能虽然不增加 low-rank 分支，但增加了新的推理开销或融合复杂度。
- 当前 smoothing 已经覆盖了这类想法中最简单的一维版本，但搜索空间可能不够强。

暂时把这一方向作为备选 idea 记录，不作为下一步优先实现目标。
