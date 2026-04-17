基于你贴出来的论文正文和 Appendix D，可以比较明确地回答这 5 个问题。

---

## 先给结论版

1. **SVDQuant 在 low-rank 分解前做了什么预处理？**  
   **有 AWQ/SmoothQuant-style 的 per-channel smoothing/scaling；没有 Hadamard rotation。**  
   顺序是：

   1. 先做 **per-channel smoothing / scaling**，把 activation outlier 迁移到 weight；
   2. 再对变换后的权重 \(\hat W\) 做 **SVD / low-rank decomposition**；
   3. 然后对 residual \(R = \hat W - L_1L_2\) 做量化。

2. **Per-channel scale 怎么算？**  
   不是 AWQ 那种“基于 activation 统计直接构造一个保护权重通道的重要性缩放”那一路，而是更接近 **SmoothQuant 的 smoothing factor**。论文 Appendix D 明确给出：
   \[
   \lambda_i = \frac{\max(|X_{:,i}|)^\alpha}{\max(|W_{i,:}|)^{1-\alpha}}
   \]
   然后：
   \[
   \hat X = X \cdot \mathrm{diag}(\lambda)^{-1}, \qquad
   \hat W = W \cdot \mathrm{diag}(\lambda)
   \]
   这样保证 \(XW = \hat X \hat W\)。

3. **Hadamard rotation 在 SVDQuant 里的角色？**  
   **论文方法里没用 Hadamard rotation。**  
   而且他们在 Section 3 还明确说了：**rotation 对 diffusion model 不适用**，因为 adaptive normalization 使得离线旋转难以和 projection weight 对齐，而在线旋转又有明显 runtime overhead。

4. **SVD 在哪个空间做？**  
   **直接在 smoothing 之后的 \(\hat W\) 上做低秩分解。**  
   公式是：
   \[
   \hat W = L_1L_2 + R
   \]
   论文里没有写 Hessian-weighted SVD。  
   但在最终实现里，**residual weight \(R\) 的量化用了 GPTQ**（Appendix D 说了 “We use GPTQ to quantize the residual weights”），所以：
   - **低秩分解本身**：不是 Hessian 加权 SVD；
   - **残差量化阶段**：可结合 GPTQ。

5. **最终部署公式是什么？**  
   主干是 **4-bit quantized residual branch**，补偿是 **16-bit low-rank branch**。部署时近似为：
   \[
   XW = \hat X \hat W = \hat X L_1L_2 + \hat X R
   \approx \hat X L_1L_2 + Q(\hat X)Q(R)
   \]
   其中：
   - \(\hat X L_1L_2\)：**16-bit low-rank branch**
   - \(Q(\hat X)Q(R)\)：**4-bit 主分支**

---

# 逐题详细回答

## 1. low-rank 分解之前具体做了哪些预处理？是 AWQ-style per-channel scaling、Hadamard rotation、还是两者都有？顺序是什么？

### 结论
**只有 per-channel smoothing/scaling，没有 Hadamard rotation。**

### 依据
论文 Section 4.2 先写：

> We first aggregate the outliers by migrating them from activation \(X\) to weight \(W\) via smoothing. Then we apply SVD to the updated weight \(\hat W\).

对应公式就是先定义：
\[
\hat X = X \cdot \mathrm{diag}(\lambda)^{-1},\qquad
\hat W = W \cdot \mathrm{diag}(\lambda)
\]
然后再分解：
\[
\hat W = L_1L_2 + R
\]

### 关于 rotation
Section 3 明确提到 rotation 是已有思路之一，但他们说：

- diffusion models 里有 **adaptive normalization**
- runtime-generated normalization weights 使得 **offline rotation** 不好做
- **online rotation** 又会带来明显运行时开销

所以 **SVDQuant 不是“scaling + Hadamard rotation + SVD”**，而是：

\[
\text{smoothing/scaling} \rightarrow \text{SVD on } \hat W \rightarrow \text{quantize residual}
\]

---

## 2. Per-channel scale 是怎么算的？是不是像 AWQ 那样基于 activation 的 per-channel 统计量，然后 \(A' = A/s, W' = \mathrm{diag}(s)W\)？

### 结论
**形式上和你写的“\(A'=A/s,\; W'=\mathrm{diag}(s)W\)”是同一路的 smoothing 变换，但具体的 scale 公式是 SmoothQuant 风格，不是 AWQ 论文里常见的那套表述。**

Appendix D 明确给出：

\[
\lambda_i = \frac{\max(|X_{:,i}|)^\alpha}{\max(|W_{i,:}|)^{1-\alpha}}
\]

其中：

- \(X \in \mathbb{R}^{b \times m}\)
- \(W \in \mathbb{R}^{m \times n}\)
- 第 \(i\) 个输入通道对应 activation 的第 \(i\) 列 \(X_{:,i}\)
- 对应 weight 的第 \(i\) 行 \(W_{i,:}\)

然后做：

\[
\hat X = X \cdot \mathrm{diag}(\lambda)^{-1}
\]
\[
\hat W = W \cdot \mathrm{diag}(\lambda)
\]

### 和你写法的关系
你写的是：

\[
A' = A/s,\qquad W' = \mathrm{diag}(s)W
\]

如果把矩阵乘法方向、权重存储 convention 对齐，本质是一样的“把 activation 某通道缩小，并把 weight 对应通道反向放大”这个思路。  
但 **SVDQuant 论文的明确写法** 是：

\[
\hat X = X \cdot \mathrm{diag}(\lambda)^{-1},\qquad
\hat W = W \cdot \mathrm{diag}(\lambda)
\]

因为他们定义的是 \(XW\)，其中 \(W \in \mathbb{R}^{m\times n}\)，所以缩放是作用在 **\(W\) 的输入通道维**，写成右乘 \(\mathrm{diag}(\lambda)\)。

### \(\alpha\) 怎么定？
不是固定的。Appendix D 说：

> It is decided offline by searching for the best migration strength \(\alpha\) for each layer to minimize the layer output mean squared error (MSE) after SVD on the calibration dataset.

所以流程是：

1. 用 calibration data 收集 \(X\)；
2. 对每层搜索 \(\alpha\)；
3. 目标是最小化 **做完 smoothing + SVD 后的 layer output MSE**。

这点很重要：  
**\(\alpha\) 的选择不是单纯为了 smoothing 本身，而是为了最终 SVDQuant 整体误差最小。**

---

## 3. Hadamard rotation 在 SVDQuant 里的角色是什么？是在 scaling 之后、SVD 之前？还是根本没用？

### 结论
**根本没用。**

### 更准确地说
论文把 rotation 当作对比背景提到，但明确说它对 diffusion model 不适合，因此 SVDQuant 没采用。

### 为什么没用？
Section 3 的逻辑是：

- 对 W4A4 来说，outlier 很麻烦；
- 传统缓解方法包括 QAT 和 rotation；
- 但 **QAT 太贵**；
- **rotation 不适用于 diffusion models**，因为 adaptive normalization 让离线旋转失效，在线旋转又有明显开销。

所以你如果问“是否是 scaling 后再 Hadamard 再 SVD”，答案是：

> **不是。论文方法链路里没有 Hadamard rotation 这一步。**

---

## 4. SVD 低秩分解是在哪个空间做的？是在经过 scaling（和/或 rotation）之后的 \(W'\) 上直接做截断 SVD，还是有 Hessian 加权？

### 结论
**是在 smoothing 后的 \(\hat W\) 上做低秩分解，论文没有描述 Hessian-weighted SVD。**

### 论文公式
Section 4.2：

\[
\hat W = L_1L_2 + R
\]

也就是说，low-rank 分解对象就是 **变换后的 weight \(\hat W\)**。

### 是否是“直接截断 SVD”？
论文叙述上是“use SVD to decompose \(\hat W\) into low-rank branch and residual”，并结合 Figure 5 讨论 \(\hat W\) 的奇异值谱前几项很大。  
所以方法意图上就是：

- 在 \(\hat W\) 的谱空间里，
- 把最大的若干奇异值成分拿出来做低秩分支，
- 剩余部分 \(R\) 更容易量化。

这基本就是 **标准截断 SVD / rank-\(r\) 近似** 的范式。

### 有没有 Hessian 加权？
**没有看到 Hessian-weighted SVD 的描述。**

但要区分两件事：

#### 1) 低秩分解阶段
没有 Hessian 加权的说法。

#### 2) residual 量化阶段
Appendix D 写了：

> We use GPTQ to quantize the residual weights.

GPTQ 本身是二阶近似/海森相关的 PTQ 方法。  
所以更准确地讲：

- **SVD 阶段：** 非 Hessian-weighted，直接对 \(\hat W\) 做低秩分解；
- **Residual quantization 阶段：** 可以用 GPTQ，因此量化时利用了二阶信息。

### 一个更贴切的流程图
\[
(X, W)
\rightarrow
(\hat X, \hat W)\text{ via smoothing}
\rightarrow
\hat W = L_1L_2 + R
\rightarrow
R \xrightarrow{\text{GPTQ / quantization}} Q(R)
\]

---

## 5. 最终的部署形式是什么？量化主分支和低秩补偿分支的具体公式

### 核心公式
论文 Section 4.2 直接给出了：

\[
XW = \hat X \hat W = \hat X L_1L_2 + \hat X R
\approx \hat X L_1L_2 + Q(\hat X)Q(R)
\]

这里：

- \(\hat X L_1L_2\)：**16-bit low-rank branch**
- \(Q(\hat X)Q(R)\)：**4-bit residual/main branch**

### 分支结构怎么理解？
设 rank 为 \(r\)，则：

- \(L_1 \in \mathbb{R}^{m\times r}\)：down projection
- \(L_2 \in \mathbb{R}^{r\times n}\)：up projection
- \(R \in \mathbb{R}^{m\times n}\)：残差权重

推理时：

1. 输入先变为 \(\hat X = X\mathrm{diag}(\lambda)^{-1}\)
2. low-rank branch：
   \[
   Y_{\text{lr}} = \hat X L_1L_2
   \]
   用 16-bit 计算
3. 主分支：
   \[
   Y_{\text{main}} \approx Q(\hat X)Q(R)
   \]
   用 4-bit 计算
4. 输出相加：
   \[
   Y \approx Y_{\text{lr}} + Y_{\text{main}}
   \]

### 从 kernel 角度
Section 4.3 / Figure 6 还说明了部署时不是把 low-rank branch 完全独立跑，而是：

- \(L_1\) 的 down projection 和 activation quantization kernel 融合；
- \(L_2\) 的 up projection 和 4-bit GEMM kernel 融合。

所以系统层面更像：

\[
\text{fused}\big(\hat X \to (\hat X L_1,\; Q(\hat X))\big)
\]
再
\[
\text{fused}\big((\hat X L_1)L_2 + Q(\hat X)Q(R)\big)
\]

以减少额外 activation 读写。

---

# 你这 5 个问题对应的“最接近源码实现”的一句话版

如果把论文写成最简实现逻辑，大概就是：

```text
1. 用 calibration data 统计每层 X 和 W
2. 搜索 alpha
3. 计算 per-channel lambda_i = max(|X[:,i]|)^alpha / max(|W[i,:]|)^(1-alpha)
4. 构造:
   X_hat = X / lambda
   W_hat = W * lambda
5. 对 W_hat 做 rank-r SVD:
   W_hat ≈ L1 L2 + R
6. 保留 L1, L2 为高精度低秩分支
7. 对 R 做 4-bit quantization（实现里可用 GPTQ）
8. 推理:
   y ≈ X_hat L1 L2 + Q(X_hat) Q(R)
```

---

# 最后给一个针对你疑问的逐条“是/否”表

| 问题 | 答案 |
|---|---|
| 分解前有 per-channel scaling 吗？ | **有** |
| 这个 scaling 是把 activation outlier 搬到 weight 吗？ | **是** |
| 公式是不是 \( \hat X = X / s,\; \hat W = W \cdot \mathrm{diag}(s)\) 这种？ | **是，本质如此** |
| scale 是像 SmoothQuant 那样由 activation/max 和 weight/max 混合出来吗？ | **是** |
| 用了 Hadamard rotation 吗？ | **没有** |
| SVD 是在原始 \(W\) 上做的吗？ | **不是，在 \(\hat W\) 上做** |
| SVD 是 Hessian-weighted 吗？ | **不是，论文没这么做** |
| residual 量化用了 GPTQ 吗？ | **是，Appendix D 明确说了** |
| 最终主分支是 quantized residual branch 吗？ | **是** |
| 最终补偿分支是 16-bit low-rank branch 吗？ | **是** |

---
