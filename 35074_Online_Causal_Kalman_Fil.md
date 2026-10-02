# KPO：Online Causal Kalman Filtering for Policy Optimization 精读笔记

> 唯一材料：提取文本 `_tools/txt/35074_Online_Causal_Kalman_Fil.txt`（3734 行，含正文、附录 A–F 与 NeurIPS 检查表）。数字均逐字取自该文件；公式因提取被打散，我按上下文重写为 LaTeX 并标注不确定处。

**一段话概括**：GRPO 类 RL 用逐 token 的重要性采样比（IS ratio）做离策略修正，但该比值在 token 级噪声极大、局部结构不一致，会放大策略梯度方差并导致训练崩溃。本文先在 Qwen3-4B 上做实证统计（窗口频率、run-length、切换频率），揭示"局部离策略偏差结构不一致"；随后提出 KPO，把期望的 IS ratio 视为沿 token 位置演化的隐状态，用一维因果 Kalman 滤波在 log 空间在线、自回归地估计（只用当前与过去 token），得到既平滑又保留段内结构的过滤比值。在数学推理基准上，Qwen3-4B + KPO-clipped 的 avg@16 在 5 个基准最优（AIME'24 37.91、AIME'25 36.87、AMC'23 87.50、MATH500 89.42、Olympiad 54.06），并显著缓解 GRPO 的熵坍缩。

---

## 一、这篇论文的一句话贡献

**把"逐 token 的重要性采样比"重新解释为一条带噪时间序列，用一维因果 Kalman 滤波在 log 空间自回归地平滑它，从而在不放弃 token 级细粒度重加权的前提下降低梯度方差、稳定 GRPO 式策略优化。** 新颖性落在"新视角＋新方法"，而非新理论：检查表第 3 项自答 `[N/A]`，论文无理论结果。

## 二、读之前需要的前置知识

1. **重要性采样比（IS ratio）与离策略修正**。回答由旧策略 $\pi_{\theta_{old}}$ 采样、却用来更新新策略 $\pi_\theta$，两者分布不同，所以每步要乘纠偏系数 $r_t=\pi_\theta/\pi_{\theta_{old}}$。完全同策略时 $r_t=1$；一旦把大 batch 拆成多个 mini-batch 依次更新，除第一个外的都是**离策略**的，$r_t$ 会偏离 1。本文关心的正是"$r_t$ 偏离 1 的程度在 token 序列上长什么样"。
2. **PPO 的 clip 与 GRPO 的组内相对优势**。PPO 把 $r_t$ 截断到 $[1-\epsilon,1+\epsilon]$，防单步更新过猛；GRPO 去掉 value 网络，对同一 prompt 采样 $G$ 个回答，用组内归一化分数作为整条序列共享的优势 $\hat A_{i,t}$。**这是理解 KPO 的前提：KPO 没动 clip，也没动优势估计，只动了 $r_t$ 这一处。**
3. **Kalman 滤波（一维标量情形）**。它是"预测—算增益—校正"的递归：先用过程模型把上一时刻后验推到下一时刻（不确定度变大），再用新观测加权修正，权重（Kalman 增益）由"预测多不确定 vs 观测多不可靠"自动决定。在本文中它负责压掉高频 spike、留下缓慢漂移，即对 IS ratio"去噪但不抹平结构"。
4. **过程噪声 $Q$ 与观测噪声 $V$（算法 1 中写作 $R$）**。$Q$ 控制隐状态允许漂移多快，$V$ 表示原始 $r_t$ 有多不可信；二者之比 $Q/V$ 决定平滑强度，是 KPO 唯一的超参数族（附录 D.3 专门做敏感性分析）。
5. **run-length / 切换频率 / 低频能量占比（LFR）**。run-length 指同一状态（Up/Down/On）连续持续多长；切换频率指相邻 token 状态改变的比例；LFR 指信号能量落在低频分量的比例。它们既是问题诊断工具（第 1 节），也是方法有效性的证据（第 4.3 节）。

## 三、研究背景与要解决的问题

**前人两条路线**（第 1、2 节）：

- **路线 A：序列级 IS ratio**。GSPO [Zheng et al., 2025b] 与 GMPO [Zhao et al., 2025] 用一个序列级标量替换所有 token 的比值，即几何平均 $r_i=\left(\prod_{t=1}^{T}r_{i,t}\right)^{1/T}=\exp\!\left(\frac1T\sum_{t=1}^T\log r_{i,t}\right)$（式 4），且 stop-gradient。作者的批评是：它 **"totally smoothing token-wise fluctuations into a single scalar, regardless of the within-sequence off-policy heterogeneity"**，即默认整条序列的离策略程度相同，而这个假设 "commonly violated"。
- **路线 B：逐 token 单独调节**。SAPO [Gao et al., 2025] 用 soft gating 按比值大小连续调制 token 级比值；ASPO [Wang et al., 2025b] 做非对称处理（收紧高风险高比值 token、其余保留学习信号）。作者的批评是它们 **"totally neglect the temporal structure of off-policy derivation across tokens within a sequence"**：逐点独立，不用 token 间时序关系。

**作者自己的实证发现**（第 2–3 页 Figure 1）：在 Qwen3-4B 上跑 GRPO，第 100 步记录 960 个样本的 token 级 IS ratio（只统计 3 个离策略 mini-batch；窗口 50，最大长度 4096），用三个统计量：窗口出现频率、run-length、切换频率（按 ratio >1 记 up、<1 记 down）。结果：离策略 token 随序列后移**越来越频繁**；寿命**很短**（run-length：Up 1.64 / Down 1.57 / On 3.53）；状态**频繁切换**（切换频率约 0.43，而 Figure 1 文字标注为 `0.41`——**两处不一致，需对照 PDF 第 2 页核实**）。结论是 **"local off-policy deviation is structurally inconsistent at the token level"**，这恰好**推翻了作者在第 3.1 节提出的"局部同质"直觉**。作者认为该不一致会 "distort the updates of policy gradients across adjacent tokens"，最终导致 GRPO 在约 200 步后 reward 下滑（Figure 3）。

## 四、方法详解（核心）

### 4.1 整体框架与数据流

```
rollout（π_old 采样 G 条回答）→ 打分得序列级优势 Â_{i,t}（不变）
        ↓
逐 token 算 log IS ratio:  z_t = log r_t = logπ_θ(a_t|s_t) − logπ_old(a_t|s_t)
        ↓
沿 t=1..T 依次跑一维 Kalman 滤波（只用 z_1..z_t）→ 后验均值 ρ̂_{t|t}
        ↓
回到比值空间:  r̃_t = exp(ρ̂_{t|t})
        ↓
把 GRPO 目标里的 r_{i,t} 换成 r̃_{i,t}，其余（clip、组内归一化优势、loss 聚合）全不变
```

核心是**因果性**：预测步只用过去 token，与自回归生成方向一致，不会看到未来（这也是不能简单用双向平滑的原因）。

### 4.2 关键公式逐个拆解

**(1) 状态空间模型（式 5）**

$$\rho_t=\rho_{t-1}+\eta_t,\quad \eta_t\sim\mathcal N(0,Q);\qquad z_t=\rho_t+\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,V)$$

$\rho_t$ 是隐状态，即"真正想要的、平滑后的 IS ratio"，在 log 空间取值；$z_t=\log r_t$ 是观测，即由新旧策略算出的原始 log 比值；$Q$ 是过程噪声方差，$V$ 是观测噪声方差（**算法 1 把同一个量记作 $R$，属记号不一致**）。设计理由：$\rho_t=\rho_{t-1}+\eta_t$ 为随机游走，作者称其 capture 了"离策略程度沿 token 缓慢演化"的直觉；观测方程把原始比值的差异全部归因于噪声。在 log 空间滤波是为了数值稳定。

**(2) 预测步（式 6）**

$$\hat\rho_{t|t-1}=\hat\rho_{t-1|t-1},\qquad P_{t|t-1}=P_{t-1|t-1}+Q$$

初始 $\hat\rho_{0|0}=0$、$P_{0|0}=P_0$；$\hat\rho_{0|0}=0$ 意为"第一个 token 被视为同策略（ratio = 1，$\log 1=0$）"。均值原地传递、方差加 $Q$，体现"多推进一步就多一分不确定"。

**(3) 新息与 Kalman 增益（式 7、8）**

$$\delta_t=z_t-\hat\rho_{t|t-1},\qquad K_t=\frac{P_{t|t-1}}{P_{t|t-1}+V}$$

$\delta_t$ 是新息（residual），即观测与预测之差；$K_t$ 在标量情形下满足 $K_t\in[0,1]$，充当**自适应步长**。预测越不确定（$P_{t|t-1}$ 大）或观测越可靠（$V$ 小），$K_t$ 越大、越信任新观测；反之越信任时间先验，从而抑制孤立 spike。这正是"保结构 vs 去噪"的自动平衡。

**(4) 更新步（式 9）**

$$\hat\rho_{t|t}=\hat\rho_{t|t-1}+K_t\delta_t,\qquad P_{t|t}=(1-K_t)P_{t|t-1}$$

第二行：纳入一次观测把后验方差压缩 $(1-K_t)$ 倍，$K_t$ 越大信息增益越大。

**(5) 回到比值空间与 KPO 目标（式 10）**

$$\tilde r_t=\exp(\hat\rho_{t|t})$$

$$J_{\text{KPO}}=\begin{cases}\mathbb E\!\left[\frac1G\sum_{i=1}^G\frac1T\sum_{t=1}^T\min\!\left(\tilde r_{i,t}\hat A_{i,t},\ \tilde r'_{i,t}\hat A_{i,t}\right)\right],\\[4pt]\mathbb E\!\left[\frac1G\sum_{i=1}^G\frac1T\sum_{t=1}^T \tilde r_{i,t}\hat A_{i,t}\right],\end{cases}\quad \tilde r'_t=\mathrm{clip}\!\left(\tilde r_t,\ 1-\epsilon^-,\ 1+\epsilon^+\right)$$

第一行是 **KPO-clipped**，第二行是 **KPO-unclipped**；$\epsilon^-,\epsilon^+$ 为非对称裁剪阈值，KPO-clipped 取 $0.0003$ 与 $0.0004$（附录 C.2 明确说这对值属于 **GSPO**，正文第 4.1 节同样用于 KPO-clipped，即 KPO 沿用 GSPO 阈值，只改作用对象为滤波后的 token 级比值）；$\hat A_{i,t}$ 是把 GRPO 的组内相对优势广播到每个 token 的结果。

**(6) 算法流程（附录 B.1, Algorithm 1）**：① 算 $z_t=\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t)$；② 对 $t=1..T$：预测 → 算 $S_t=P_{t|t-1}+R$、$K_t=P_{t|t-1}/S_t$ → 更新；③ 取 $\tilde r_t=\exp(\hat\rho_{t|t})$；④ 按式 (10) 算 $J_{\text{KPO}}$。**整个滤波是 $O(T)$ 的逐 token 标量递归，无额外网络、无额外可学习参数。**

### 4.3 相较基线的改动点具体在哪一处

**只改了 GRPO 目标中乘在优势上的那个系数**：原始 token 级 $r_{i,t}=\pi_\theta/\pi_{old}$ → 滤波后的 $\tilde r_{i,t}=\exp(\hat\rho_{t|t})$。相对 GRPO，KPO 让当前 token 的比值**吸收过去 token 的信息**；相对 GSPO/GMPO，它保留逐 token 差异，只把这些差异约束在低频连续形态上，而非压成标量并 stop-gradient；相对 SAPO/ASPO，它引入了时间维度的信息传递。论文用偏差-方差权衡解释合理性：GSPO 换来稳定性但偏向偏差（抹掉段间异质性），KPO 试图同时压低方差并保留结构。

## 五、实验怎么做的

**模型与数据**：Qwen3-4B [Yang et al., 2025a]；训练语料取自 DAPO [Yu et al., 2025] 的监督式 RL 语料（多样数学题＋可验证解＋奖励信号）；二值规则奖励（成功 1、失败 0）。

**基准与指标**：AIME'24、AIME'25、AMC'23、MATH500 [Hendrycks et al., 2021]、OlympiadBench，另加 Table 1 中的 **Minerva**（共 6 列；但第 4.1 节的 "five competitive benchmarks" 只列 5 个，且**未给出 Minerva 定义**，属交代缺失）。每题采样 16 个解，报告 `avg@16`（16 个样本的平均准确率）与 `pass@16`（至少一个样本正确的题目比例）。

**基线**：token 级 GRPO、序列级 GMPO、GSPO。统一训练配置：batch size 32、minibatch 8（4 个 mini-batch，只有第 1 个同策略，后 3 个离策略）、group size 8、max length 4096、actor lr $1\times10^{-6}$、loss 聚合 "sequence-mean-token-mean"、评测 top_p=1.0、temperature=1.0、H100。KPO 滤波参数：KPO-clipped $Q=1\times10^{-6}$（$V=1$），KPO-unclipped $Q=1\times10^{-4}$（$V=1$）。基线裁剪：GRPO $\epsilon=0.2$、GMPO $\epsilon=0.4$（原文写作 `ϵ = 0,4`，逗号是小数点提取错误）、GSPO $\epsilon^-=0.0003,\epsilon^+=0.0004$。

**主要结果（Table 1，逐字）**：KPO-clipped 在 5 个基准 avg@16 最优——AIME'24 **37.91**、AIME'25 **36.87**、AMC'23 **87.50**、MATH500 **89.42**、Olympiad **54.06**；并在 AIME'25（**60.00**）、MATH500（**94.80**）、Olympiad（**66.27**）的 pass@16 最优。相对最强基线 GSPO（AIME'24 32.70/60.00，AIME'25 29.16/50.00），KPO-clipped 提升到 37.91/63.33 与 36.87/60.00。KPO-unclipped 取得 AIME'24 最佳 pass@16 **66.67** 与 Minerva 最佳 avg@16 **39.15**。**Minerva 是 KPO 唯一未领先 pass@16 的基准**（GMPO 50.73 略高于 50.36）；AMC'23 的 pass@16 KPO-clipped 与 GSPO 并列 **95.00**。Table 1 中 Minerva 列数值（GRPO 38.67/50.36、GMPO 38.48/50.73、GSPO 37.17/47.79、KPO-unclipped 39.15/50.36、KPO-clipped 38.23/48.16）在提取文本中排成竖列，**列头对应关系建议对照 PDF 表 1 复核**。

**训练动力学（Figure 3）**：GRPO 在约 200 步后发散，且早期熵就坍缩到接近 0；KPO 与 GMPO 保持较高熵、小幅波动，GSPO 稳定在较低但非零水平；GMPO 与 GRPO 的 clip fraction 始终接近 0（约 **0.0015**），因其裁剪界更宽；KPO 比 GSPO **裁得更少**，作者解释为滤波后的比值局部结构一致、只有部分 token 触发裁剪，而序列级比值一旦越界就是整条序列被裁。

**分析实验一：过滤前后的 token 级统计（Table 2，逐字）**

| 指标 | Before KF | After KF |
|---|---|---|
| Token 占比 (Up/Down/On) | 0.25 / 0.22 / 0.53 | 0.35 / 0.43 / 0.22 |
| Run-length (Up/Down/On) | 1.64 / 1.57 / 3.53 | 119.95 / 135.12 / 35.11 |
| 切换频率 | 0.43（高） | 0.01（低） |
| LFR | 0.12（低） | 0.98（高） |
| 全局 / 窗口局部方差 | 0.19 / 0.15 | 1e-4 / 1e-5 |

即：滤波把"每几个 token 就跳变一次"的高频信号变成"上百 token 的连续段"，能量几乎全在低频，局部方差几乎归零。注意 Up/Down 的 run-length 已达 119.95 与 135.12，接近整条序列的长度尺度。On 占比反而从 0.53 降到 0.22，作者称这是"更紧的同策略判定带（$r\in[1-\epsilon^-,1+\epsilon^+]$）"所致，并主张不代表变差（附录 D.1）。

**分析实验二：$Q/V$ 敏感性（附录 D.3，Figure 4）**。测试 $Q/V\in\{10^{-6},10^{-4},10^{-2}\}$：越小平滑越强。$10^{-6}$ 最终 reward 最高且全程稳定上升；$10^{-4}$ 略差；$10^{-2}$ 明显更低且中后期退化（"weak filtering introduces excessive noise"）。clip fraction 方面文本称 $10^{-4}$ 最高、$10^{-6}$ 中等稳定、$10^{-2}$ 反而偏低；但**同一段结尾又把结论写成 "With stronger filtering (smaller $Q/R$)... excessively large $Q/R$"，把 $V$ 写成 $R$**，属记号混用，**需对照 PDF 核实**。

## 六、优点与贡献

1. **新视角（问题刻画层）**：把 token 级 IS ratio 当时间序列看，用三个可复现统计量把"局部结构不一致"量化出来，这是 GSPO/GMPO/SAPO/ASPO 忽略的维度。
2. **新方法且极轻量（新方法层）**：不引入可学习参数、不改损失聚合与裁剪机制，只插入一个 $O(T)$ 标量递归，因此 "compatible with existing pipelines"，工程迁移成本极低，优于需要额外 critic 或额外 rollout 的方案。
3. **可验证的中间指标（新分析层）**：不只报最终分数，还给出 run-length 从 1.6 升到约 120、切换频率从 0.43 降到 0.01、LFR 从 0.12 升到 0.98，使"滤波确实改变了比值结构"这一主张可被独立检验。
4. **稳定性证据一致**：reward、熵、clip fraction、policy gradient loss 四条曲线共同支持"GRPO 不稳定、GSPO/GMPO 早饱和、KPO 稳定且持续提升"。
5. **主动报告负面结果**：指出 Minerva 是唯一未领先的基准，并给出机制解释（滤波平滑但不移除真正极端的离策略 token，仍需 clip 兜底）。

## 七、局限与可疑之处

**作者自承**：① **无法并行**——滤波必须沿 token 顺序递归，"difficult to parallelize ... like the autoregressive generation of tokens"（结论）；② 两个平滑参数需逐场景调（附录 F，附录 D.3 建议 MoE 或 train-inference mismatch 场景调大 $Q$）；③ **无理论保证**（检查表第 3 项 `[N/A]`），**无误差棒/显著性**（第 7 项自答 `[No]`）。

**我认为可疑或没讲清楚的地方**：

1. **默认超参与"保留局部结构"的叙事自相矛盾**。取 $Q=10^{-6}$、$V=1$，稳态 Riccati 方程 $P^2+QP-QV=0$ 给出 $P^\*\approx\sqrt{QV}=10^{-3}$，稳态增益 $K^\*=P^\*/(P^\*+V)\approx10^{-3}$，等效时间常数约 $1/K^\*=1000$ 个 token，等效观测噪声标准差 $\sqrt{1/K^\*}\approx32$。这解释了 Table 2 中 run-length 上百、局部方差降到 $10^{-5}$，也意味着**默认设置下滤波几乎是整条序列的滑动平均**，与"保留段内局部结构"存在张力。**论文能反驳我的两点**：(a) Table 2 显示全局方差也降到 $1e{-}4$；(b) 若 KPO 真退化为"每序列一个标量"，它与 GSPO 的差别应只剩阈值，而 AIME 上高出约 5–8 分。所以更准确的质疑是：**收益来自 token 级结构，还是仅仅来自"比 GSPO 的硬几何平均更软的平滑"？论文没有这个对照。**
2. **缺少最简单的对照基线**。没有把 **EMA / 中值滤波 / 固定窗口滑动平均 / 双向平滑** 作为基线。若无状态量的 EMA（$\tilde r_t=\alpha r_t+(1-\alpha)\tilde r_{t-1}$）在同等平滑强度下效果相当，"Kalman 状态空间＋新息＋自适应增益"这套机制的必要性就受挑战。这是我认为最值得补的实验。
3. **与已有 Kalman+GRPO 工作的关系未讨论**。参考文献含 Wang et al., 2025a（KRPO，"Kalman filter enhanced GRPO"），但它既不是基线，正文也未解释差异，只被归入"clarify the role of filtering effects and smoothing in group-based advantage estimation"。对主打 Kalman 的论文，这是**基线公平性与相关工作覆盖的明显缺口**。
4. **只在一个模型、一个规模上验证**：全部主实验仅用 Qwen3-4B，"at scale" 的说法缺 7B/14B/32B 或多模型族支撑，而 GRPO 的不稳定恰恰是规模相关现象。
5. **对照不干净**：KPO-clipped 与 KPO-unclipped 使用不同 $Q$（$10^{-6}$ vs $10^{-4}$），故"clipped 普遍强于 unclipped"**混杂了滤波强度差异**，无法干净归因于 clip。
6. **无多种子、无误差棒**。AIME 约 30 题、AMC 约 40 题（由 pass@16 的 92.50/95.00/87.50 这种 2.5 步长可推），**单次运行 3–5 分差距完全可能落在随机波动内**，而作者使用了 "definitely validates its superiority" 的过强措辞。
7. **叙事内在矛盾**：第 3.1 节动机是"局部同质"，第 2 页实证却是"局部结构不一致、counter 了我们的直觉"，而方法（随机游走先验＋强平滑）恰恰是为"局部同质"服务的。更诚实的表述应是：KPO 把观测到的跳变**当作噪声**处理，而非"保结构"。
8. **缺成本报告**：只说 "lightweight"，未给相对 GRPO 的 wall-clock/吞吐，也未说明递归是 CUDA kernel 还是 Python 循环——这对"能否用于 scale"很关键。
9. **On 状态判定规则在过滤前后不同**（过滤前 $r=1$，过滤后 $r\in[1-\epsilon^-,1+\epsilon^+]$），因此"On 占比 0.53→0.22"受规则变化影响，仅由附录 D.1 一句话解释。

## 八、与同批论文的关系

本批 10 篇（按文件名核实的全称）：1008_SCA（Spatial Credit Assignment, GUI agent）、15751_HGPO（Hierarchy-of-Groups Policy Optimization, ICLR 2026）、18_PA-MoE（Phase-Aware Mixture of Experts for Agentic RL）、22669_GraphGPO（Graph-Based Credit Assignment for Agentic RL）、35074_KPO（本篇）、1406_CITA（Choosing Before Acting: Comparative Value Estimation）、14071_AgentBrew（Offline Tool-Use Agent Learning）、1423_CIPO（Counterfactual Imagination Policy Optimization）、1432_PhGPO（Pheromone-Guided Policy Optimization）、13085_AAWM（Agent-Authored World Modeling）。

- **与 15751_HGPO、22669_GraphGPO、1008_SCA：竞争性最强，且方法论互补**。四篇都建立在 **GRPO 类组内优势目标**上，但改的"旋钮"不同：KPO 改 **IS ratio（沿 token 时间维）**；HGPO 改 **优势的层次粒度（面向长程 agent 任务）**；GraphGPO 改 **信用分配的图结构（超越轨迹级、做步级归因）**；SCA 改 **优势估计的几何条件（利用 rollout 的坐标/bbox 结构）**。因此 KPO 与它们**可叠加**：把滤波比值插进 GraphGPO 或 SCA 的目标，理论上可同时获得步级/空间级信用与时序平滑的离策略修正；反过来它们也是最直接的竞争者——评审时会被问"同样是稳定性或信用分配改进，为什么在比值维度做"。
- **与 1423_CIPO、1432_PhGPO**：同为策略优化层方法，但问题设定是**工具使用/长程规划**，改动点在探索与技能粒度而非离策略比值，属**互补**。
- **与 1406_CITA、14071_AgentBrew、13085_AAWM**：属工具使用/世界模型与离线学习一支，与 KPO 的数学推理设定几乎不重叠，**互补且距离较远**。值得注意的对照是：CITA 需额外 rollout 估计中间决策的下游影响、AgentBrew 用单批原始轨迹做离线学习，二者都强调**不增加在线交互**来获得更细监督——这与 KPO"不加任何额外网络与计算、只在已有比值上在线滤波"的哲学同源，可作 related work 的共同动机。
- **与 18_PA-MoE**：论文指出 MoE 路由不连续会扰乱 IS ratio（引 Zheng et al., 2025b），并在附录 D.3 建议"使用 MoE 时调大 $Q$"；PA-MoE 改的是专家路由粒度。二者在"MoE 场景下稳住训练"上**可直接组合**。
- **提醒**：KPO 的直接前作 **GSPO / GMPO / SAPO / ASPO / KRPO 均不在本批 10 篇内**，写综述时须另行补充。

## 九、如果我要接着做，可以做什么

**方向 1：证明收益来自"结构"而非"更软的平滑"——加无状态对照基线。**
- *改什么*：实现三个同预算基线：(a) EMA $\tilde r_t=\alpha r_t+(1-\alpha)\tilde r_{t-1}$；(b) 固定窗口滑动平均；(c) 双向（非因果）平滑。把 (a)(b) 的强度扫到使 token 局部方差与 KPO 相当。
- *为什么有意义*：若 EMA 在同方差水平下成绩接近 KPO，说明真正起作用的是"降低比值方差"而非因果状态空间建模；反之则是 KPO 的有力加固。两种结果都可发表。
- *第一步*：Qwen3-4B + AIME'24/AIME'25 训练 800 步，画"局部方差 vs avg@16"曲线，把 KPO/EMA/滑窗三者工作点标在同一张图上。

**方向 2：让滤波强度自适应——把 $Q/V$ 从全局常数变成按序列估计。**
- *改什么*：用观测残差在线估噪，例如以滑动窗口内 $\delta_t$ 的方差估计 $V$、以序列前若干 token 的漂移幅度估计 $Q$；再给 MoE 或 train-inference mismatch 场景一个自动调大 $Q$ 的规则。
- *为什么可能有效*：附录 D.3 自己证明 $Q/V$ 从 $10^{-6}$ 到 $10^{-2}$ 结果差异巨大，而默认对所有序列都用 $10^{-6}$（等效窗长约 1000 token），对短回答与长回答显然不合理；自适应有望去掉这个最脆弱的超参数。
- *第一步*：统计 batch 内序列长度分布与 $\delta_t^2$ 分布，画出"最优固定 $Q/V$ vs 序列长度"曲线；若呈单调关系，先按长度分桶设置 $Q/V$ 作为改进版基线。

**方向 3：把"沿 token 因果滤波"推广到 agent 的"沿步因果滤波"。**
- *改什么*：在步级/图级信用分配方法（如本批 GraphGPO、HGPO）中，对每一步的 IS ratio 也做因果 Kalman 滤波，状态空间从"token 索引"换成"交互步索引"，优势换成步级优势。
- *为什么可能有效*：KPO 的假设（局部同质、全局异质）在 agent 轨迹上更自然——同一子任务内连续若干步的同策略程度通常相近，跨子任务差异大；且 agent 轨迹更长、更易失稳。
- *第一步*：在 GraphGPO 或 HGPO 的开源代码上只插入滤波模块（不改优势与图构建），在 WebShop/ALFWorld 一类长程环境对比"有/无滤波"，并报告 run-length 与切换频率以确认机制被触发。

## 十、自测问题

1. 用式 (4) 说明 GSPO/GMPO 的序列级比值与 KPO 的滤波比值在"序列内差异"上的本质不同；为什么作者认为前者牺牲了 within-sequence 异质性而后者没有？（提示：对比 stop-gradient 与 token 间信息流。）
2. 写出 Kalman 增益 $K_t$，说明 $Q$ 增大时 $K_t$ 如何变化；为什么附录 D.3 中 $Q/V=10^{-2}$ 反而 reward 更低、clip fraction 却偏低？请给出你自己对作者所称 "counterfactual" 现象的解释。
3. 若去掉 KPO 的因果性（允许 $\hat\rho_{t|t}$ 使用 $z_{t+1:T}$ 做双向平滑）会有什么后果？作者为何坚持只用过去 token？该约束与 LLM 自回归生成有何对应？
4. Table 2 中滤波后 On 占比从 0.53 降到 0.22，是否意味着"更多 token 变成离策略、训练更不稳"？请结合 run-length、切换频率与附录 D.1 的 On 判定规则（$r\in[1-\epsilon^-,1+\epsilon^+]$）回答，并指出该比较中**判定规则变化**带来的混淆。
5. 作为审稿人，指出本文实验设计中最影响结论可信度的两个问题，并设计一个**单卡可完成**的最小补充实验来回应其中之一（说明模型、基准、对照设置与要观察的指标）。
