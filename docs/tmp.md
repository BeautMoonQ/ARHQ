你现在这版已经把 **核心想法、目标函数、闭式解、实验信号** 讲清楚了；作为内部技术文档已经很完整。但如果目标是 **arXiv tech report / 正式论文先行版**，建议重点优化以下几个方面：

---

# 1. 先把论文主线压得更清楚

现在稿子信息量很大，但有些地方像“研究笔记”，正式 tech report 可以把叙事主线收紧成：

> **Activation quantization error is not isotropic. ARHQ measures its covariance and removes the weight directions that amplify this error most, by assigning them to a high-precision low-rank branch.**

可以在 Introduction 里明确三句话：

1. 现有 low-rank quantization methods 主要看权重本身，例如 SVDQuant 近似 \(W\)。
2. 但在 aggressive activation quantization 下，真正决定输出误差的不只是 \(W\) 的能量，而是 activation quantization residual \(E_x\) 如何经过 \(W\) 传播。
3. ARHQ 因此用 \(E_x^\top E_x\) 构造 residual Hessian，在这个度量下做 weighted low-rank decomposition。

也就是说，论文最核心的 claim 可以写得更锋利：

```text
ARHQ is not a weight reconstruction method.
ARHQ is an activation-residual-aware weight splitting method.
```

这个定位非常重要，因为它可以避免被审稿人理解成“只是 SVD 前乘了一个 covariance matrix”。

---

# 2. 建议统一符号，尤其是误差符号

现在文中同时用了：

```text
Q_x(X) = X + Delta_x
```

和：

```text
E_x = X - Q_x(X)
```

于是：

```text
E_x = - Delta_x
```

这在推导上没错，因为平方范数不受符号影响，但读者容易困惑。

建议统一成一种定义。比如全篇都用：

```text
E_x = Q_x(X) - X
E_w = Q_w(W) - W
```

那么量化输出误差是：

```text
\hat{Y} - Y
= E_x W^T + X E_w^T + E_x E_w^T
```

如果你更喜欢：

```text
E_x = X - Q_x(X)
```

也可以，但要在第一次引入时明确：

```text
We define the activation residual as the negative quantization error:
E_x = X - Q_x(X).
The sign is irrelevant for the quadratic objective.
```

正式论文里建议避免同时使用 `Delta_x` 和 `E_x` 表示同一个东西的相反数。

---

# 3. Objective 部分可以上升为一个 Proposition

你现在的推导是正确的，但正式论文里建议把 weighted low-rank solution 写成一个明确命题。

例如：

## Proposition: ARHQ closed-form solution

Given \(W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}\) and residual covariance

\[
G_x = \frac{1}{N} E_x^\top E_x,
\]

consider

\[
\min_{\operatorname{rank}(L) \le r}
\left\|
(W - L)G_x^{1/2}
\right\|_F^2.
\]

If \(G_x\) is positive definite, let

\[
M = W G_x^{1/2}
\]

and let

\[
M_r = U_r \Sigma_r V_r^\top
\]

be the best rank-\(r\) approximation of \(M\). Then an optimal solution is

\[
L^\star = M_r G_x^{-1/2}.
\]

Equivalently, with

\[
B = U_r \Sigma_r,
\qquad
A = G_x^{-1/2} V_r,
\]

we have

\[
L^\star = B A^\top.
\]

这里要注意一个细节：如果写成 \(A = G_x^{-1/2} C\)，那么你需要让 \(C = V_r\)。如果你写：

```text
M ~= B C^T
```

那么最好明确：

```text
B = U_r Sigma_r
C = V_r
```

否则读者可能不知道奇异值到底放在 \(B\)、\(C\) 还是二者均分。

---

# 4. 需要处理 \(G_x\) 半正定或者病态的情况

实际中：

\[
G_x = E_x^\top E_x / N
\]

通常是半正定，甚至可能 rank deficient。尤其当：

- calibration token 数量小于 \(D_{\text{in}}\)；
- 某些通道 activation residual 很小；
- quantizer 对部分通道几乎无误差；
- 使用 smoothing 后 residual covariance 更低秩；

那么 \(G_x^{-1/2}\) 会不稳定。

你现在用了：

```text
lambda_i <- clamp(lambda_i, eps)
```

这是工程上合理的，但正式论文里建议说明这相当于优化一个 regularized metric：

\[
G_{x,\epsilon} = U \operatorname{diag}(\max(\lambda_i, \epsilon)) U^\top.
\]

然后目标实际变成：

\[
\min_{\operatorname{rank}(L) \le r}
\left\|
(W - L)G_{x,\epsilon}^{1/2}
\right\|_F^2.
\]

建议加一段：

```text
In practice, \(G_x\) may be singular or ill-conditioned. We therefore use a regularized residual Hessian \(G_{x,\epsilon}\). This has two effects: it stabilizes the inverse square root, and it prevents unconstrained growth of \(L\) along directions with near-zero residual energy.
```

这个很重要，因为如果 \(G_x\) 有零特征值，目标函数对这些方向不惩罚，\(L\) 在 null space 上并不唯一。clamp 之后才有唯一、稳定的解。

---

# 5. 需要更清楚地区分三种 Hessian / covariance

你现在说：

```text
G_x = E_x^T E_x / N
```

不是普通：

```text
H_x = X^T X / N
```

这一点很好。但可以进一步强化，因为审稿人可能会问：

> 为什么叫 Hessian？它只是 covariance 吧？

建议加一段解释：

对于目标：

\[
\mathcal{L}(W_{\text{res}})
=
\frac{1}{N}
\left\|
E_x W_{\text{res}}^\top
\right\|_F^2
=
\operatorname{Tr}
\left(
W_{\text{res}} G_x W_{\text{res}}^\top
\right),
\]

它关于 \(W_{\text{res}}\) 的 Hessian 是 block-diagonal 的，每个输出通道对应同一个输入侧矩阵 \(G_x\)。因此 \(G_x\) 可以被看作这个 residual propagation loss 的 input-side Hessian。

可以写成：

\[
\nabla_{W_{\text{res}}} \mathcal{L}
=
2 W_{\text{res}} G_x,
\]

\[
\nabla^2_{W_{\text{res}}} \mathcal{L}
=
2 I_{D_{\text{out}}} \otimes G_x.
\]

这样 “Residual Hessian” 这个名字就更站得住。

---

# 6. 需要明确 ARHQ 优化的不是完整真实误差

你在 Current Limitations 里已经提到了，但建议更早一点提醒读者：

真实推理是：

\[
\hat{Y}
=
Q_x(X) Q_w(W_{\text{res}})^\top
+
X L^\top.
\]

而 ARHQ 的闭式目标对应的是：

\[
\hat{Y}_{\text{approx}}
=
Q_x(X) W_{\text{res}}^\top
+
X L^\top.
\]

也就是说，ARHQ 是针对 activation quantization residual 的一阶子问题，而不是完整 joint quantization objective。

建议在 Objective 前后加一句：

```text
We first isolate the activation-residual propagation term and solve it exactly. Weight quantization of \(W_{\text{res}}\) is then applied after the split, and its interaction with the split is studied empirically.
```

这样会降低审稿人对“为什么不直接优化全量误差”的质疑。

---

# 7. 需要强调低秩分支使用的是原始高精度 activation

推理公式里：

\[
\hat{Y}
=
Q_x(X) Q_w(W_{\text{res}})^\top
+
X L^\top
\]

意味着主分支用 \(Q_x(X)\)，低秩分支用 \(X\)。

这在理论上很自然，但工程上读者会问：

1. 推理时是否还保留 bf16/fp16 activation？
2. 低秩分支的 \(X A\) 是 bf16 GEMM 还是 fp16 GEMM？
3. 如果主路径中 activation 量化是在线完成的，那么 \(X\) 在量化前确实可得吗？
4. 低秩分支是否带来额外 memory bandwidth 和 latency？

建议加一个 Implementation Assumptions 小节：

```text
ARHQ assumes that the pre-quantized activation \(X\) is available at the linear operator boundary. The main branch consumes the quantized activation \(Q_x(X)\), while the low-rank branch computes \(X A\) and \((X A)B^\top\) in higher precision. This is similar to LoRA-style side branches and introduces an additional cost of \(O(N r (D_{\text{in}} + D_{\text{out}}))\) multiply-adds per linear layer.
```

并给出参数开销：

\[
\text{extra params}
=
r(D_{\text{in}} + D_{\text{out}}).
\]

相对原始权重：

\[
\frac{r(D_{\text{in}} + D_{\text{out}})}
{D_{\text{in}}D_{\text{out}}}.
\]

如果是 square layer \(D_{\text{in}} = D_{\text{out}} = D\)，则：

\[
\frac{2r}{D}.
\]

例如 \(D = 2560\)、\(r=128\)，额外参数比例约：

\[
\frac{256}{2560} = 10\%.
\]

这类数字很有帮助。

---

# 8. Smoothing 部分建议改成矩阵形式，避免歧义

现在写：

```text
X_s = X / s
W_s = W * s
```

读者可以理解，但正式论文建议写成对角矩阵：

令：

\[
S = \operatorname{diag}(s).
\]

那么：

\[
X_s = X S^{-1},
\]

\[
W_s = W S.
\]

于是：

\[
X_s W_s^\top
=
X S^{-1} (W S)^\top
=
X S^{-1} S W^\top
=
X W^\top.
\]

这比 `W * s` 更不容易误解。

低秩分支也建议写成：

\[
L_s = B A_s^\top,
\]

\[
W_{\text{res},s} = W_s - L_s.
\]

推理：

\[
\hat{Y}
=
Q_x(XS^{-1})
Q_w(W_{\text{res},s})^\top
+
(XS^{-1} A_s)B^\top.
\]

另外最好说明 smoothing scale \(s\) 是怎么得到的：

- SmoothQuant-style？
- grid search？
- 根据 activation max 和 weight max？
- learnable？
- per-channel？
- 是否和 ARHQ 联合优化？

如果现在没有成熟方法，也可以说：

```text
In this report we use an existing smoothing rule and treat smoothing as a preprocessing step. Jointly optimizing smoothing and ARHQ is left for future work.
```

---

# 9. Quantization Adaptivity 部分可以更强，但需要避免过度 claim

你现在说 ARHQ 能适配 block-wise int4、NVFP4 等，这是合理的，因为 \(E_x\) 是直接测量的。

但最好补一句边界：

```text
This adaptivity is empirical and calibration-dependent. If the calibration distribution does not match deployment, \(G_x\) may misestimate the true residual structure.
```

否则审稿人会质疑 calibration overfitting。

也可以加一个观点：

ARHQ 适配的是：

```text
quantizer + calibration distribution + layer input distribution
```

而不是单纯适配 quantizer。

更准确地说：

\[
G_x
=
\mathbb{E}_{x \sim \mathcal{D}_{\text{calib}}}
\left[
(x - Q_x(x))(x - Q_x(x))^\top
\right].
\]

所以它依赖：

1. 激活分布；
2. 量化器；
3. block size；
4. scale policy；
5. calibration data；
6. smoothing transform。

---

# 10. Relation to SVDQuant 需要更严谨

你写：

```text
SVDQuant: min ||W - L||_F^2
ARHQ: min ||(W - L) G_x^{1/2}||_F^2
```

这个对比非常清楚。

但如果正式论文里提 SVDQuant，建议注意两点：

1. 不要把 SVDQuant 简化得太过头，除非你确认其完整方法确实如此。
2. 可以把它称为 “SVD-style weight decomposition baseline” 或 “Frobenius SVD baseline”，而不是直接完整定义 SVDQuant。

更安全的写法：

```text
A standard SVD-based low-rank split minimizes the unweighted Frobenius reconstruction error of \(W\). In contrast, ARHQ replaces the Euclidean metric with the activation-residual metric \(G_x\).
```

如果你要正式比较 SVDQuant，最好在 Related Work 里准确介绍其完整 pipeline。

---

# 11. 实验部分需要补全 baseline 和设置，否则说服力不够

现在 SNR 和 ZebraLogic 结果很有价值，但作为 arXiv report 还需要更多实验细节。

建议补充以下内容。

## 11.1 量化设置

需要写清楚：

- activation quantization format：NVFP4？block size？
- weight quantization format：NVFP4？AWQ int4？per-channel？per-group？
- scale 计算方式；
- zero-point 是否使用；
- symmetric / asymmetric；
- fake quant 还是真实 kernel；
- accumulation dtype；
- low-rank branch dtype；
- \(W_{\text{res}}\) 是否重新校准 scale；
- \(L\) 是否 bf16 保存；
- \(A\)、\(B\) 是否量化；
- smoothing scale 的 dtype 和保存方式。

否则别人很难复现，也很难判断提升来自哪里。

## 11.2 校准数据

需要说明：

- calibration dataset；
- calibration token 数；
- sequence length；
- batch size；
- 是否用模型生成 prompt；
- 是否只用 prefill；
- 是否覆盖所有层；
- \(G_x\) 是按 layer 收集还是按 module 收集；
- q/k/v/o 是否共享 calibration；
- attention 和 FFN 是否都做。

## 11.3 SNR 定义

建议明确 SNR 公式：

\[
\operatorname{SNR}(Y, \hat{Y})
=
10 \log_{10}
\frac{\|Y\|_F^2}
{\|Y - \hat{Y}\|_F^2}.
\]

或者如果你用的是 per-token/per-layer average，也要说明。

同时要说明 baseline 是什么：

- Baseline 是原始 NVFP4？
- Improvement over baseline 的 baseline 值是多少？
- 每层 SNR 是先平均再算 dB，还是先算 dB 再平均？
- q/k/v/o average 是四类 projection 等权平均，还是按元素数加权？

建议表格里直接加一列 baseline：

| Scope | Baseline | SVD Raw | ARHQ Raw | SVD Smooth | ARHQ Smooth |
|---|---:|---:|---:|---:|---:|

这样读者不用从 improvement 反推。

## 11.4 增加 rank ablation

rank=128 是一个点，但正式报告最好给：

| Rank | SVD Raw | ARHQ Raw | SVD Smooth | ARHQ Smooth |
|---:|---:|---:|---:|---:|
| 32 | ... | ... | ... | ... |
| 64 | ... | ... | ... | ... |
| 128 | ... | ... | ... | ... |
| 256 | ... | ... | ... | ... |

ARHQ 的优势如果在多个 rank 下都稳定，会非常有说服力。

尤其要看：

- 低 rank 下 ARHQ 是否优势更大；
- 高 rank 下是否收敛到类似结果；
- smoothing 是否和 rank 有交互。

## 11.5 增加 layer-wise plots

建议画：

1. 每一层 q/k/v/o 的 SNR；
2. ARHQ - SVDQuant 的 layer-wise gain；
3. smoothing 前后的 gain；
4. residual Hessian spectrum；
5. \(WG_x^{1/2}\) 的 singular value spectrum vs \(W\) 的 singular value spectrum。

这些图能非常直观地支持你的 claim：

```text
ARHQ selects different low-rank directions from vanilla SVD.
```

尤其是最后一个图很关键。你可以展示：

- \(W\) 的 top singular vectors；
- \(W G_x^{1/2}\) 的 top singular vectors；
- 二者 subspace similarity。

例如：

\[
\left\|
U_{\text{SVD}}^\top U_{\text{ARHQ}}
\right\|_F^2 / r.
\]

如果相似度不高，就说明 ARHQ 真不是普通 SVD。

---

# 12. 需要加入 ARHQ 的消融实验

建议至少加这些 ablation：

## 12.1 \(G_x\) vs \(H_x\)

比较：

\[
G_x = E_x^\top E_x / N
\]

和：

\[
H_x = X^\top X / N.
\]

也就是：

| Metric | Objective |
|---|---|
| Weight SVD | \(\|W-L\|_F^2\) |
| Activation Hessian | \(\|(W-L)H_x^{1/2}\|_F^2\) |
| Residual Hessian | \(\|(W-L)G_x^{1/2}\|_F^2\) |

这个消融非常重要，因为它可以证明：

```text
不是任意 Hessian weighting 都有效，关键是 activation quantization residual Hessian。
```

## 12.2 Random residual Hessian

用打乱的 \(G_x\)、diagonal-only \(G_x\)、identity 做对比：

- full \(G_x\)；
- diagonal \(G_x\)；
- block-diagonal \(G_x\)；
- shuffled diagonal；
- identity。

这可以说明：

1. 通道间相关性是否重要；
2. 只用 per-channel residual variance 是否已经足够；
3. full covariance 的收益是否值得计算成本。

## 12.3 Calibration size ablation

比较 calibration tokens：

- 128；
- 512；
- 2k；
- 8k；
- 32k。

看 \(G_x\) 稳定性和下游指标。

## 12.4 Quantizer adaptivity ablation

用同一个 \(W\) 和 calibration data，对不同 quantizer 构造不同 \(G_x\)：

- uniform int4；
- block-wise int4；
- NVFP4；
- FP8；
- maybe int8 activation。

然后测试：

1. 用对应 quantizer 的 \(G_x\)；
2. 交叉使用其他 quantizer 的 \(G_x\)。

如果对应 \(G_x\) 最好，就能证明 adaptivity。

---

# 13. ZebraLogic 评测需要更谨慎表述

你已经写了“早期阶段，样本数只有 140”，这是对的。

但正式报告里建议更进一步：

1. 明确这不是最终 benchmark。
2. 报告置信区间。
3. 不要过度强调“与 bf16 持平”，因为 130/140 样本下方差很大。

可以用 binomial standard error 粗略说明。

对于 puzzle accuracy：

\[
p = 130/140 = 0.9286.
\]

标准误差约：

\[
\sqrt{p(1-p)/140}
\approx 0.0218.
\]

95% 区间大约是：

\[
\pm 4.3\%.
\]

所以 130/140 和 127/140 的差异未必显著。

建议文字改成：

```text
On this small 140-puzzle subset, ARHQ achieves the highest observed accuracy among quantized variants and matches the bf16 run. However, the difference between 127/140 and 130/140 is within the expected statistical noise of such a small sample. We therefore treat this result as an encouraging signal rather than conclusive evidence.
```

这会显得更专业。

---

# 14. 需要增加 end-to-end 性能指标

如果是量化 tech report，只给 SNR 和 accuracy 还不够。最好补：

1. perplexity；
2. MMLU / GSM8K / MATH-500 / HumanEval / GPQA / BBH 等；
3. throughput；
4. latency；
5. memory footprint；
6. KV cache 影响；
7. prefill latency 和 decode latency 分开；
8. low-rank branch 对 kernel fusion 的影响。

ARHQ 可能会有额外开销，所以需要主动回答：

```text
How much accuracy do we recover per additional FLOP / parameter?
```

可以给一个表：

| Method | Weight bits | Act bits | Extra rank | Extra params | Prefill tok/s | Decode tok/s | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|

哪怕目前没有真实 kernel，也可以先写：

```text
This report focuses on numerical quality. Kernel-level latency evaluation is left for future work.
```

但最好不要完全不提。

---

# 15. 需要说明 ARHQ 对哪些层使用

现在 SNR 是 36 层 attention projection，ZebraLogic 有 `arhq_r128_all`。建议明确：

- only q/k/v/o？
- FFN gate/up/down？
- lm_head？
- embedding？
- all linear layers？
- rank 是否所有层相同？
- rank 是否按层自适应？
- smoothing 是否所有层使用？

如果 attention-only 和 all-layer 的效果不同，建议分别报告。

正式报告中可以定义：

```text
ARHQ-attn: apply ARHQ to q_proj, k_proj, v_proj, o_proj.
ARHQ-ffn: apply ARHQ to gate_proj, up_proj, down_proj.
ARHQ-all: apply ARHQ to all transformer linear layers except embeddings and lm_head.
```

---

# 16. 可以考虑加入 rank allocation 方法

现在统一 rank=128。正式论文可以提出一个自然 extension：

根据每层 residual propagation energy 分配 rank。

例如 layer/module 的未校正误差：

\[
\mathcal{E}_0
=
\|E_x W^\top\|_F^2
=
N \operatorname{Tr}(W G_x W^\top).
\]

rank-\(r\) 后剩余误差为：

\[
\mathcal{E}_r
=
\sum_{i > r} \sigma_i^2(WG_x^{1/2}).
\]

因此 rank \(r\) 带来的收益是：

\[
\Delta_r
=
\sum_{i=1}^r \sigma_i^2(WG_x^{1/2}).
\]

可以用这个做全模型 rank budget allocation：

```text
Given a total low-rank budget, allocate ranks to layers with the largest marginal singular-value gains.
```

这会让 ARHQ 从“一个分解方法”升级成“一个全模型 compression framework”。

---

# 17. 可以增加对角近似版本，降低成本

完整 \(G_x \in \mathbb{R}^{D_{\text{in}} \times D_{\text{in}}}\) 的特征分解成本是：

\[
O(D_{\text{in}}^3).
\]

对于大模型大维度可能贵。

建议增加 practical variants：

## Full ARHQ

\[
G_x = E_x^\top E_x / N.
\]

## Diagonal ARHQ

\[
G_x = \operatorname{diag}
\left(
E_x^\top E_x / N
\right).
\]

这时：

\[
G_x^{1/2}
\]

只是 per-channel scale，计算很便宜。

## Block-diagonal ARHQ

按照 quantization block 或 hidden chunks 做 block covariance。

你可以在报告中说：

```text
The current experiments use full covariance for maximum quality. Diagonal and block-diagonal ARHQ are promising lower-cost variants.
```

如果你能加一组 diagonal vs full 的实验，会很好。

---

# 18. 需要检查 weighted decomposition 的数值稳定性

潜在风险：

\[
A = G_x^{-1/2} C
\]

如果某些 \(\lambda_i\) 很小，\(A\) 的范数可能很大。即使目标上这些方向 residual 很小，工程上也可能造成：

1. 低秩分支 activation \(XA\) 爆炸；
2. bf16 overflow 或精度差；
3. \(W_{\text{res}} = W - BA^\top\) 变大；
4. \(W_{\text{res}}\) 更难量化。

你已经在 limitations 提到 \(W_{\text{res}}\) 可能更难量化。建议更具体地把这个归因到 inverse Hessian step：

```text
The inverse square-root mapping may amplify directions with small residual eigenvalues. Although these directions are weakly penalized by the ARHQ objective, they can affect the dynamic range of \(L\), \(A\), and \(W_{\text{res}}\). We therefore clamp eigenvalues and optionally normalize the LoRA factors.
```

可考虑加入 factor balancing：

如果：

\[
L = B A^\top,
\]

可以重新缩放：

\[
B' = B R,
\qquad
A' = A R^{-\top},
\]

保持 \(L\) 不变，但改善 \(XA\) 和 \(B\) 的数值范围。

一个简单做法是列归一化：

\[
\alpha_j = \sqrt{\frac{\|A_j\|_2}{\|B_j\|_2}},
\]

然后调整 \(A_j, B_j\) 的 scale。

这个作为 implementation detail 会很有帮助。

---

# 19. 当前 limitation 可以展开成 Future Work

你现在的 limitation 很关键。建议把 Future Work 分成三类：

## 19.1 Weight-quantization-aware ARHQ

目标：

\[
\min_L
\left\|
E_x(W-L)^\top
\right\|_F^2
+
\lambda
\left\|
X((W-L)-Q_w(W-L))^\top
\right\|_F^2.
\]

可以说没有闭式解，考虑：

- alternating optimization；
- STE fine-tuning；
- post-split correction；
- quantizer-aware rank selection；
- constrain \(W_{\text{res}}\) dynamic range。

## 19.2 Joint smoothing and ARHQ

目前 smoothing 是预处理。未来可以优化：

\[
\min_{s,L}
\left\|
E_s(W_s-L_s)^\top
\right\|_F^2
+
\text{weight quantization penalty}.
\]

## 19.3 Kernel-aware ARHQ

低秩分支引入额外 GEMM，未来需要：

- fused kernels；
- rank-specific kernels；
- batching low-rank branches；
- decode-time optimization；
- maybe quantize \(A\) or \(B\)。

---

# 20. 可以加入 Algorithm box

正式 tech report 建议放一个算法框。

例如：

```text
Algorithm 1: ARHQ decomposition for one linear layer

Input:
  Weight W
  Calibration activations X
  Activation quantizer Q_x
  Target rank r
  Regularization epsilon

1. X_q = Q_x(X)
2. E_x = X - X_q
3. G = E_x^T E_x / N
4. Eigendecompose G = U diag(lambda) U^T
5. lambda = max(lambda, epsilon)
6. G_sqrt = U diag(sqrt(lambda)) U^T
7. G_inv_sqrt = U diag(1 / sqrt(lambda)) U^T
8. M = W G_sqrt
9. Compute truncated SVD:
     M_r = U_r Sigma_r V_r^T
10. B = U_r Sigma_r
11. A = G_inv_sqrt V_r
12. W_res = W - B A^T

Output:
  W_res, A, B
```

注意这里第 11 步：

\[
A = G^{-1/2} V_r
\]

然后：

\[
B A^\top
=
U_r \Sigma_r V_r^\top G^{-1/2}
\]

因为 \(G^{-1/2}\) 对称。

---

# 21. Abstract 和 Introduction 可以提前写

建议你现在就加一个 abstract，哪怕是草稿。比如：

```text
We propose Activation Residual Hessian Quantization (ARHQ), a post-training weight splitting method for low-bit activation-weight quantization. Instead of approximating the weight matrix in the Euclidean Frobenius norm, ARHQ constructs a residual Hessian from the measured activation quantization error and decomposes each linear layer under this metric. The resulting high-precision low-rank branch absorbs weight directions that most amplify activation quantization residuals, while the remaining residual weight is quantized in the main branch. The ARHQ decomposition admits a closed-form solution via the SVD of \(WG_x^{1/2}\), where \(G_x\) is the covariance of activation quantization residuals. On Qwen3-4B-Thinking-2507 attention projections, ARHQ improves layer-wise SNR over SVD-based low-rank splitting, and preliminary ZebraLogic evaluation suggests improved generation quality under NVFP4 quantization.
```

然后 Introduction 可以按：

1. Problem；
2. Observation；
3. Method；
4. Results；
5. Contributions。

Contributions 可以写：

```text
Our contributions are:
1. We identify activation quantization residual propagation as a major error source in low-bit linear layers.
2. We propose the activation residual Hessian \(G_x = E_x^\top E_x/N\) and formulate low-rank weight splitting under this metric.
3. We derive a closed-form ARHQ decomposition through the truncated SVD of \(WG_x^{1/2}\).
4. We show that ARHQ is naturally quantizer-adaptive because \(G_x\) is measured from the actual activation quantizer.
5. We provide preliminary SNR and downstream generation results showing improvements over SVD-style low-rank splitting.
```

---

# 22. 建议改名或补充缩写解释

“Activation Residual Hessian Quantization” 这个名字可以，但有一点可能引起误解：

- 你不是直接 quantize Hessian；
- 你也不是 Hessian quantization；
- 你是 Hessian-aware decomposition for quantization。

如果保留 ARHQ，建议第一次出现时写：

```text
Activation Residual Hessian Quantization (ARHQ) is a Hessian-aware low-rank weight splitting method for quantized linear layers.
```

也可以考虑副标题：

```text
ARHQ: Activation-Residual-Hessian-Aware Low-Rank Splitting for Low-Bit Quantized Transformers
```

这样比单独 “Quantization” 更准确。

---

# 23. 推荐调整后的章节结构

你现在的内容可以重排成：

```text
1. Introduction
2. Background and Error Decomposition
   2.1 Quantized linear layer
   2.2 Activation and weight quantization errors
   2.3 Low-rank side branch

3. Activation Residual Hessian
   3.1 Activation residual covariance
   3.2 Difference from activation Hessian
   3.3 Residual propagation loss

4. ARHQ Decomposition
   4.1 Objective
   4.2 Closed-form solution
   4.3 Numerical regularization
   4.4 Algorithm

5. Smoothing Extension
   5.1 Equivalent smoothing transform
   5.2 ARHQ in smoothed space

6. Practical Implementation
   6.1 Calibration
   6.2 Quantizer-specific residual collection
   6.3 Storage and compute overhead
   6.4 Inference formula

7. Experiments
   7.1 Setup
   7.2 Layer-wise SNR
   7.3 Rank ablation
   7.4 Smoothing ablation
   7.5 Downstream evaluation

8. Discussion
   8.1 Why residual Hessian instead of activation Hessian
   8.2 Quantizer adaptivity
   8.3 Interaction with weight quantization

9. Limitations and Future Work
10. Conclusion
```

---

# 24. 最值得优先补的内容

如果时间有限，我建议按优先级补：

## Priority 1

1. 统一符号；
2. 加 Proposition 和闭式解证明；
3. 解释 \(G_x\) 为什么叫 Hessian；
4. 说明 \(G_x\) 奇异时的 regularization；
5. 明确 low-rank branch 使用高精度 \(X\)；
6. 补充 SNR 定义和 quantization setting。

## Priority 2

1. rank ablation；
2. \(G_x\) vs \(H_x\) vs identity ablation；
3. layer-wise SNR plot；
4. calibration size ablation；
5. overhead table。

## Priority 3

1. ZebraLogic full set；
2. multi-seed；
3. MATH-500 / GSM8K / perplexity；
4. latency / throughput；
5. diagonal or block-diagonal ARHQ variant。

---

# 25. 总体评价

这版最强的地方是：

1. **动机很清楚**：不是重建 \(W\)，而是降低 activation residual 传播。
2. **目标函数合理**：\(\|E_x(W-L)^\top\|_F^2\) 直接来自误差分解。
3. **闭式解漂亮**：SVD of \(WG_x^{1/2}\) 很简洁。
4. **和 SVDQuant 的差异明确**：Euclidean weight metric vs activation residual metric。
5. **初步实验信号不错**：ARHQ raw 已经优于 SVD raw，smoothing 后进一步提升。

目前主要短板是：

1. 工程细节和复现实验设置不够完整；
2. 对 \(G_x\) 的 Hessian 解释还可以更数学化；
3. 对半正定/病态 \(G_x\) 的处理需要正式化；
4. 下游评测样本较小，需要更谨慎；
5. 缺少 ablation 来证明“residual Hessian”而不是“一般 Hessian/SVD weighting”带来收益。

如果把这些补上，这篇 tech report 的完成度会明显接近 arXiv 先行版。






@article{li2024svdquant,
  title={Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models},
  author={Li, Muyang and Lin, Yujun and Zhang, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
  journal={arXiv preprint arXiv:2411.05007},
  year={2024}
}

@article{lin2023awq,
  title={Awq: Activation-aware weight quantization for llm compression and acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  journal={arXiv preprint arXiv:2306.00978},
  year={2023}
}

@article{frantar2022gptq,
  title={Gptq: Accurate post-training quantization for generative pre-trained transformers},
  author={Frantar, Elias and Ashkboos, Saleh and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2210.17323},
  year={2022}
}

@article{xiao2022smoothquant,
  title={Smoothquant: Accurate and efficient post-training quantization for large language models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  journal={arXiv preprint arXiv:2211.10438},
  year={2022}
}

@article{park2026serq,
  title={Serq: Saliency-aware low-rank error reconstruction for llm quantization},
  author={Park, Yeonsik and Kim, Hyeonseong and Choi, Seungkyu},
  journal={arXiv preprint arXiv:2603.08185},
  year={2026}
}

@article{ashkboos2024quarot,
  title={Quarot: Outlier-free 4-bit inference in rotated llms},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L. and Li, Bo and Cameron, Pashmina and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}

@article{liu2024spinquant,
  title={Spinquant: Llm quantization with learned rotations},
  author={Liu, Zechun and Zhao, Changsheng and Fedorov, Igor and Soran, Bilge and Choudhary, Dhruv and Krishnamoorthi, Raghuraman and Chandra, Vikas and Tian, Yuandong and Blankevoort, Tijmen},
  journal={arXiv preprint arXiv:2405.16406},
  year={2024}
}







Algorithm 1  ARHQ Quantize W given calibration activations X and rank r.

    X_q ← Q_x(X)                                      // quantize activations
    E_x ← X_q − X                                     // activation residuals
    G_x ← E_x^T @ E_x / N                             // residual Hessian

    U, λ ← Eig(G_x)                                   // eigendecomposition
    λ ← clamp(λ, eps, +∞)                             // stabilize small eigenvalues

    G_x^{1/2}  ← U @ diag(sqrt(λ)) @ U^T              // metric square root
    G_x^{-1/2} ← U @ diag(1 / sqrt(λ)) @ U^T          // inverse metric square root

    M ← W @ G_x^{1/2}                                 // weighted weight matrix
    U_r, Σ_r, V_r ← SVD_r(M)                          // rank-r truncated SVD

    lora_B ← U_r @ Σ_r                                // output low-rank factor
    lora_A ← G_x^{-1/2} @ V_r                         // input low-rank factor

    L ← lora_B @ lora_A^T                             // high-precision branch
    W_res ← W − L                                     // residual main-branch weight
    W_q ← Q_w(W_res)                                  // quantized residual weight

    return W_q, lora_A, lora_B