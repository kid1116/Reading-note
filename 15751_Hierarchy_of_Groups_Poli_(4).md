# HGPO 精读笔记

论文标题：**Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks**（ICLR 2026）
作者：Shuo He, Lang Feng, Qi Wei, Xin Cheng, Lei Feng, Bo An（南洋理工大学 / 东南大学）
代码：https://github.com/langfengQ/verl-agent/tree/master/recipe/hgpo

> 本文针对长时程 LLM 智能体 RL 中 stepwise 分组策略优化的隐患——**历史上下文不一致（historical context inconsistency）**：同一步级分组内各步当前状态相同，但记忆模块里的历史不同，使组内相对优势估计有偏、策略优化劣化。作者提出 HGPO：按历史上下文的公共长度把每一步归入多个层级分组，再自适应加权融合各层优势。ALFWorld 上把 Qwen2.5-1.5B 的分布内成功率从 GiGPO 的 90.16% 提到 92.77%（K=2），显存与训练时间几乎不变。

## 一、这篇论文的一句话贡献

**它明确指出并实证了"步级分组里历史上下文不一致会造成优势估计偏差"这一问题，然后用"按历史公共长度分层建组 + 按层自适应加权融合优势"这一处改动，在不加模型、不加 rollout 的前提下，把有偏的步级优势换成偏差与方差折中更好的估计量。**

## 二、读之前需要的前置知识

**1. 长时程任务与稀疏延迟奖励。** LLM 智能体连续多轮交互：第 $t$ 轮观测 $s_t$、输出文本动作 $a_t$。本文设定是**只有轨迹最后一步给标量奖励**（成功 10 / 失败 0），轨迹为 $\tau=\{(s_1,a_1),\dots,(s_T,a_T)\}$。这就是信用分配的根源难题：知道"这局成了"，不知哪一步有功。

**2. trajectory-wise 与 stepwise 两种框架。** 前者把整条轨迹拼成长 prompt $\pi_\theta(a_t\mid s_{0:t},x)$，上下文随轮数膨胀（context explosion）；后者把轨迹拆成独立的步，用**记忆模块只保留最近 $K\ll T$ 个历史上下文**，prompt 长度稳定。本文方法全部建立在 stepwise 之上——这正是"上下文不一致"的来源。

**3. group-based RL 与组相对优势。** PPO 需额外 critic 估优势；GRPO 家族直接用同一 prompt 下 $N$ 条轨迹（一个 group）的奖励统计算优势。本文把单轮 GRPO 适配到 stepwise，得轨迹级优势 $A^T(\tau_i)$（式 1）。基线 GiGPO 再进一步：把**所有组轨迹中当前状态相同**的步聚成步级组 $\tilde{G}_{\tilde{s}_i}$，算 $A^S(\tilde{s}_i)$（式 2）。关键点：**"当前状态相同"不等于"prompt 相同"**，因为记忆里的历史不同。

**4. 偏差–方差折中。** 估计量既要偏差小（接近真实优势 $A^*$）又要方差小（训练稳定）。粗粒度分组样本多、方差小，但整条轨迹共享一个值、偏差大；Oracle 分组上下文一致、偏差小，但样本太少、方差大。全文动机就是在这个折中里找更好的点。

**5. Oracle 步与 step utilization ratio。** Oracle 步 = 当前状态**且**历史上下文都完全相同的步，其优势估计被当作最准参考。作者用 Oracle 组做 pilot 实验（图 2），论证"上下文不一致 → 偏差"；Table 2 的 step utilization ratio 是各层分组能覆盖的步占比。

## 三、研究背景与要解决的问题

**前人做法。** 单轮任务上 RLOO、GRPO、DAPO、Clip-Cov、GSPO 很成功，但都是为单轮设计的。搬到多轮时 RAGEN、Search-R1 采用 trajectory-wise 框架，问题是**有效上下文长度随轮数快速膨胀，导致严重的 context explosion**。于是近期研究转向 stepwise 框架（Feng et al. 2025b 即 GiGPO、Luo et al. 2025c、Chen et al. 2025b、Team 2025、Yu et al. 2025b、Wang et al. 2025c），用记忆模块保留历史。

**具体局限。** stepwise 框架下，即便同一环境任务，"共享同一当前状态"的 step-level anchor group 内各步的记忆内容仍可能不同，有效 prompt 因此不同，组相对优势会**有偏**。GiGPO 的 $G_{\tilde{s}_i}$ 只匹配当前状态，正是这一局限的直接受害者。

**作者的论证。** 两步：(1) 直观例子：图 1(b) 用两条组轨迹 $\tau_1,\tau_2$ 说明状态 $s_2$（紫色）处的步级组含历史不一致的步，并主张"相对优势应在相同当前状态**且**相同历史上下文下计算"；(2) pilot 实证：以 Oracle 组为参照，图 2(a)(b) 显示轨迹级与步级优势都有明显偏差，**轨迹级偏差远大于步级**；但"只用 Oracle 步"被图 2(c)(d) 否掉——Oracle 步占比低、其平均组规模小，会抬高方差、破坏稳定性。

**定位一句话**：问题不是"分组不够细"，而是"分组条件漏掉了历史上下文这一维"；解法是补上这一维，同时用分层结构避免"只用 Oracle 步"带来的样本枯竭。

## 四、方法详解（核心）

### 4.1 整体框架与数据流

HGPO 沿用 stepwise group-based RL 的外壳（多轮 rollout、critic-free、单一 actor），**唯一实质改动集中在优势估计环节**。流程：

1. **Rollout**：采样任务 $x\sim p(X)$，用同一初始状态 $s_0$ 初始化 $N$ 个环境（$N=8$），并行跑 $T$ 轮得到 $\{\tau_i\}$（图 3 画了 4 条）。
2. **分组（离线、纯 hashmap 查表）**：对每步 $(s_t^{(i)},a_t^{(i)})$，用"当前状态 + 最近 $k$ 步历史"整套键查表，得第 $k$ 层同伴集合，$k=0,\dots,K$。$k=0$ 就是 GiGPO 的步级组。
3. **多组优势计算**：每层 $k$ 各用组内奖励统计算出 $A_k^H$。
4. **自适应加权融合**：按 $w_k$ 线性组合成最终 $A^H$。
5. **策略更新**：把 $A^H$ 塞进 clip 版 GRPO 目标 $J_{\text{HGPO}}(\theta)$，迭代 160 轮。

关键工程性质：**分组完全离线**，只依赖已有 rollout 的 hashmap 查表，不需额外模型或数据采集——这是"零额外 rollout"的原因。

### 4.2 关键公式逐个拆解

**(1) 轨迹级优势（基线，式 1）**

$$A^T(\tau_i)=\frac{R(\tau_i)-\frac{1}{|G_\tau|}\sum_{j\in G_\tau}R(\tau_j)}{\sigma_{G_\tau}}$$

$R(\tau_i)$ 是轨迹总回报；$G_\tau$ 是同一任务下的轨迹组；$\sigma_{G_\tau}$ 是组内奖励标准差。分子是"比组平均好多少"，分母归一化。**同一轨迹内所有步共享这一个值**——这是它粗粒度、偏差大的原因。

**(2) 步级优势（GiGPO，式 2）** $A^S(\tilde{s}_i)=\dfrac{R(\tilde{s}_i)-\frac{1}{|\tilde{G}_{\tilde{s}_i}|}\sum_{j\in\tilde{G}_{\tilde{s}_i}}R(\tilde{s}_j)}{\sigma_{\tilde{G}_{\tilde{s}_i}}}$（提取文本中式 2 的求和项排版顺序有错位，表达式本身一致）。$\tilde{G}_{\tilde{s}_i}$ 是所有轨迹中"当前状态相同"的步集合。它更细，但**未约束历史上下文**，仍有偏。

**(3) $k$-步上下文算子（式 3）**

$$\mathcal{C}^k\!\left(s_t^{(i)}\right)=\begin{cases}\left(s_{t-k}^{(i)},s_{t-k+1}^{(i)},\cdots,s_t^{(i)}\right), & t\ge k\\[4pt]\left(s_0^{(i)},s_1^{(i)},\cdots,s_t^{(i)}\right), & t<k\end{cases}$$

$k\in[0,K]$，返回当前状态连同前面 $k$ 个历史状态的元组；$t<k$（轨迹开头历史不够）时退化为从 $s_0$ 起。它是所有分组的"键函数"。

**(4) 第 $k$ 层层级分组（式 4）**

$$G^H_k\!\left(s_t^{(i)}\right)=\left\{(j,n)\in I:\ \mathcal{C}^k\!\left(s_t^{(i)}\right)=\mathcal{C}^k\!\left(s_n^{(j)}\right)\right\},\quad I=\{(i,t)\mid 1\le i\le N,\ 1\le t\le T\}$$

$I$ 是全部 (轨迹, 步) 索引。含义：**只有当前状态与前 $k$ 步历史都相同的步才同组**，$k$ 越大越苛刻。

**(5) 层级嵌套（式 5）**

$$G^H_0\!\left(s_t^{(i)}\right)\supseteq G^H_1\!\left(s_t^{(i)}\right)\supseteq\cdots\supseteq G^H_K\!\left(s_t^{(i)}\right),\qquad \left|G^H_0\right|\ge\cdots\ge\left|G^H_K\right|$$

（提取文本此处排版成竖排方括号，已按语义复原。）**匹配更长历史 = 取更小子集**，"层级"二字的全部含义在此。$k=0$ 退化为 GiGPO 的步级分组，$k=K$ 对应 Oracle 组。分层带来两点好处：提升步利用率（低层大组仍接纳高层落单的步）、降低优势估计方差。

**(6) 第 $k$ 层优势（式 6）**

$$A^H_k\!\left(s_t^{(i)}\right)=\frac{R\!\left(s_t^{(i)}\right)-\frac{1}{\left|G^H_k\right|}\sum_{(j,n)\in G^H_k}R\!\left(s_n^{(j)}\right)}{\sigma_{G^H_k}}$$

结构与式 1、2 同构，只把参照组换成第 $k$ 层组。步级回报采用 GiGPO 的折扣累积：$r_t^{(i)}=\sum_{j=t}^{T}\gamma^{\,j-t}r_j^{(i)}$，$\gamma\in(0,1]$（实验取 0.95）。**这一步关键**：它把最终稀疏奖励向前摊到每步，使"当前状态相同"的步之间可比——越早到达成功终局的状态折扣回报越高。

**(7) 自适应加权融合（式 7，核心创新点）**

$$A^H\!\left(s_t^{(i)}\right)=\sum_{k=0}^{K}w_k\,A^H_k\!\left(s_t^{(i)}\right),\qquad w_k=\frac{(k+1)^{\alpha}}{\sum_{k}(k+1)^{\alpha}},\quad \alpha\ge 0$$

权重单调递增：**层级越高（历史越一致）权重越大**；$\alpha$ 控制分布尖锐程度。$\alpha=0$ 退化为均匀权重（消融 "w/o Ada. $w_k$"）。实现上**优势为零的组被跳过**；默认 $\alpha=1$。

**(8) 策略优化目标（式 8）**

$$J_{\text{HGPO}}(\theta)=\mathbb{E}\Big[\tfrac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T}\min\!\big(\rho_\theta(a_t^{(i)})A^H(s_t^{(i)}),\ \mathrm{clip}(\rho_\theta(a_t^{(i)}),1\pm\epsilon)A^H(s_t^{(i)})\big)-\beta D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right)\Big]$$

$$\rho_\theta\!\left(a_t^{(i)}\right)=\frac{\pi_\theta\!\left(a_t^{(i)}\mid s_t^{(i)},x\right)}{\pi_{\theta_{\text{old}}}\!\left(a_t^{(i)}\mid s_t^{(i)},x\right)}$$

$\rho_\theta$ 是重要性采样比，$\epsilon$ 是 clip 参数，$\beta$ 控 KL 强度（实验 0.01），$\pi_{\text{ref}}$ 是参考策略。**这就是 GRPO 的 clip 目标，只把优势从 $A^T$ 换成 $A^H$**——"改动只有一处"的最直白体现。

### 4.3 理论分析：偏差–方差折中（Proposition 4.1）

记第 $k$ 层偏差、方差为 $b_k=\mathrm{Bias}[A^H_k]$、$v_k=\mathrm{Var}[A^H_k]$，真实优势 $A^*$。两条假设：(1) 偏差递减 $B_T\ge b_0\ge(b_1,\dots,b_{K-1})\ge b_K\ge0$；(2) 方差递增 $v_0\le(v_1,\dots,v_{K-1})\le v_K\le V_T$。其中 $B_T,V_T$ 为轨迹级估计的偏差与方差，$b_0,v_0$ 为步级估计的。合理性辩护：组轨迹数（8）通常小于步级组规模，故轨迹级偏差与方差都更大；$K$ 增大时 $G^H_k$ 变小，方差上升。

在 $\sum_k w_k=1$、$w_k\ge0$、且各层估计不相关（$\mathrm{Cov}(A^H_k,A^H_{k'})=0,\ k\ne k'$）下：

$$\mathrm{Bias}[A^H]=\sum_{k=0}^{K}w_k b_k,\qquad \mathrm{Var}[A^H]=\sum_{k=0}^{K}w_k^2 v_k$$

由 $b_k$ 递减、$\sum w_k=1$ 得

$$b_K\le \mathrm{Bias}[A^H]\le b_0\le B_T$$

（提取文本此处公式丢空、只剩碎片，上式按附录 B 推导复原，建议对照 PDF 第 6、17–18 页核实。）方差侧由 Cauchy–Schwarz 有 $\frac{1}{K+1}\le\sum_k w_k^2\le1$，故

$$\frac{v_0}{K+1}\le \mathrm{Var}[A^H]\le v_K\le V_T$$

**结论**：HGPO 在偏差和方差上都在步级（$k=0$）与 Oracle（$k=K$）之间插值，取得更好折中。附录 B 另有三者对比表：粗粒度为 $B_T,V_T$，细粒度为 $b_0,v_0$，Hierarchy-of-Groups 为加权组合且两项均下降。

### 4.4 训练流程（Algorithm 1）

**输入**：$\pi_{\theta_{\text{old}}}$、任务分布 $p(X)$、折扣因子 $\gamma$、权重 $\omega$、clip 参数 $\epsilon$、KL 系数 $\beta$、组规模 $N$、历史长度 $K$、权重指数 $\alpha$。每次迭代：

1. 同步旧策略 $\theta_{\text{old}}\leftarrow\theta$。
2. 采样 $x\sim p(X)$，初始化 $N$ 个**完全相同**的环境（起点一致是分组有意义的前提）。
3. 对 $t=1..T$：采样 $a_t^{(i)}\sim\pi_{\theta_{\text{old}}}(\cdot\mid s_t^{(i)},x)$，执行后得 $\{r_t^{(i)}\}$ 与 $\{s_{t+1}^{(i)}\}$。
4. **分组**：按式 (5) 做上下文感知层级分组。
5. **优势计算**：按式 (7) 组内算优势并加权融合。
6. **策略更新**：最大化 $J_{\text{HGPO}}(\theta)$。

### 4.5 相较基线的改动点具体在哪

- **相对 GRPO**：GRPO 只用式 (1) 的轨迹级优势，全轨迹共享一个标量；HGPO 换成按历史分层的步级优势融合。改动只在**优势估计**，策略目标形式（clip + KL）不变。
- **相对 GiGPO（最直接对手）**：GiGPO 的分组只用"当前状态相同"这一个条件（即仅 $k=0$ 层）。HGPO 精确地改两处：(a) 分组键从 $s_t$ 扩展为 $\mathcal{C}^k(s_t)$，并同时保留 $k=0..K$ 全部层级；(b) 增加式 (7) 把多层优势融合为一个标量。**$K=0$ 时 HGPO 严格退化回 GiGPO 的步级分组**——论文明确写了这点，消融 "w/o $G^H_1:K$" 测的就是它。
- **不加东西**：无新模型、无新 rollout、无 critic；额外成本仅 hashmap 查表。

## 五、实验怎么做的

**环境。** ALFWorld（文本具身，4,639 个任务实例，六类：Pick & Place / Examine in Light / Clean & Place / Heat & Place / Cool & Place / Pick Two & Place，最多 50 步）与 WebShop（模拟 HTML 购物网站，超 110 万商品、12k 条用户指令，最多 30 步）。ALFWorld 报分布内（In-Success）与分布外（Out-Success）成功率；WebShop 报平均任务得分与平均任务成功率。

**基座与基线。** Qwen2.5-1.5B-Instruct 与 Qwen2.5-7B-Instruct。基线：(1) 闭源 GPT-4o、Gemini-2.5-Pro；(2) prompting：Qwen2.5 直接提示、ReAct、Reflexion；(3) RL：PPO（带 critic）、RLOO、GRPO、GiGPO。

**公平性与超参。** 所有 RL 方法共享同一套超参：group size $N=8$、每组 16 个 rollout（共 $16\times8=128$ 个环境，PPO 用 128 个独立环境）、actor lr 1e-6（critic 1e-5，仅 PPO）、rollout 温度 1.0、验证温度 0.4、KL 系数 0.01、$\gamma=0.95$、规则奖励（成功 10 / 失败 0 / 非法动作 -0.1）；ALFWorld mini-batch 256、最大 prompt 2048(4096) token、最大回复 512 token；WebShop mini-batch 64、最大 prompt 4096 token。HGPO 的 $\alpha=1$。1.5B 用 2 张 H100、7B 用 4 张，均 160 次迭代，验证集 512，评测 3 个随机种子。

**主要结果（Table 1，数字逐字取自提取文本）。**

*Qwen2.5-1.5B-Instruct：*

| 方法 | ALFWorld In-Suc. | Out-Suc. | WebShop Task Sco. | Task Suc. |
|---|---|---|---|---|
| ReAct (prompting) | 12.8 | – | 40.1 | 11.3 |
| PPO (with critic) | 54.4±3.1 | – | 73.8±3.0 | 51.5±2.9 |
| GRPO | 72.8±3.6 | 70.1±2.5 | 75.8±3.5 | 56.8±3.8 |
| GiGPO (K=2) | 90.16±0.22 | 84.76±2.83 | 84.95±2.57 | 66.53±1.92 |
| **HGPO (K=2)** | **92.77±1.08** | **90.16±0.78** | **85.56±2.86** | **71.54±4.00** |
| GiGPO (K=4) | 93.29±1.07 | 91.53±1.99 | 86.80±1.60 | 73.24±2.25 |
| **HGPO (K=4)** | **94.85±0.92** | **92.12±1.63** | **90.64±1.05** | **78.12±2.06** |

*Qwen2.5-7B-Instruct：* GiGPO (K=2) 93.29±0.40 / 92.18±0.39 / 88.93±1.49 / 77.60±1.68 对 **HGPO (K=2) 95.44±0.62 / 92.05±0.22 / 88.96±1.04 / 78.51±1.40**；GiGPO (K=4) 95.63±0.49 / 95.18±0.98 / 89.51±0.59 / 77.79±0.92 对 **HGPO (K=4) 95.96±0.49 / 94.87±1.24 / 88.49±0.41 / 79.29±1.17**。作者概括"最高到 95.96% 的 ALFWorld 分布内成功率与 79.29% 的 WebShop 任务成功率"。

**作者的三条结论性观察。** (i) RL 训练全面优于 prompting，说明 RL 对这些长时程任务必要；(ii) **小模型收益更大**：平均增益 3.41% 对 0.74%（K=2）、2.72% 对 0.13%（K=4），解释是小模型 agentic 能力弱、步更长更冗余、优势偏差更大（这两个"平均增益"的口径原文未说明，需对照 PDF 第 7 页核实）；(iii) 增大 $K$ 对 GiGPO 与 HGPO 都有提升，说明方法随记忆长度可扩展。

**进一步分析。**

- **分层组规模分布（图 4，1.5B，第 160 epoch）**：0-context 组比 1/2-context 组有更大比例的大组（它忽略历史）；$K$ 增大时质量从大组向小组迁移，说明共享完整历史的 Oracle 步通常只形成小组、方差更高。
- **step utilization ratio（Table 2，1.5B）**：ALFWorld，$K=2$ 时 0-C/1-C/2-C = 0.97/0.75/0.52；$K=4$ 时 0.98/0.77/0.54/0.34/0.19。WebShop，$K=2$：0.92/0.64/0.44；$K=4$：0.90/0.59/0.40/0.21/0.09。**几乎全部步都落在 0-context 组**，层级越高利用率越低——这是"Oracle 步稀缺"的量化证据。
- **计算预算（Table 3，1.5B，ALFWorld）**：公共部分耗时 297.8 / 288.6 / 284.7 / 282.5 秒（第 40/80/120/160 epoch）；查表时间 GRPO 0.048 / 0.027 / 0.018 / 0.013 秒、GiGPO 0.126 / 0.080 / 0.055 / 0.034 秒、HGPO 0.693 / 0.601 / 0.456 / 0.246 秒；峰值显存 GiGPO 0.1035 / 0.0967 / 0.0605 / 0.0391 MB、HGPO 0.1393 / 0.1229 / 0.1078 / 0.0406 MB。作者结论：HGPO 比 GRPO、GiGPO 平均多花约 0.425 s 与 0.472 s，**占总执行时间不到 0.001%**。
- **参数分析（Table 4，$\alpha$）**：$K=2$ 时 $\alpha=0$ 较好（ALFWorld In-Suc. 94.33±1.08、Task Suc. 73.95±4.21），$K=4$ 时 $\alpha=0$ 反降（92.12±0.40 / 73.82±3.26）；$\alpha=1$ 与 $\alpha=2$ 在 $K$ 从 2 增到 4 时都上升（$\alpha=1$：92.77→94.85、71.54→78.12；$\alpha=2$：90.36→93.55、72.00→75.39）。$\alpha=2$ 比 $\alpha=1$ 略降，归因于高层组样本少、方差大。默认取 $\alpha=1$。
- **消融（Table 5，1.5B）**：① "**w/o $G^H_1:K$**"（去掉层级，只用 $G^H_0$，即 GiGPO 式做法）：ALFWorld 明显退化，"约 2.8% 的分布内成功率与 5.8% 的分布外成功率"（表中 K=4 为 91.14/88.80，对 HGPO 的 94.85/92.12）；WebShop 类似，但 $K=2$ 时略升。② "**w/o Ada. $w_k$**"（均匀权重 $\alpha=0$）：$K=2$ 时两环境都略升，$K=4$ 时下降。③ "**w Eq. (1)**"（把轨迹级优势并入式 7）：除 $K=2$ 的 WebShop 外几乎全部下降，因轨迹级优势"高偏差、低方差"。④ **只用 Oracle 步做策略优化的实验"failed"**。
- **附录**：A 为伪代码；B 为 Proposition 4.1 完整推导；C 为环境、对比方法、超参与 prompt（`<think>`/`<action>` 两段式，history length 为 2）；D 报告**旧版 Verl-agent** 下的结果（Table 7，如 1.5B 上 GiGPO K=4 为 85.15±2.81 / 80.98±0.45 / 88.5±0.49 / 74.08±0.98，HGPO K=4 为 92.45±0.81 / 89.06±2.34 / 88.90±0.90 / 75.91±1.19）与训练动态曲线（图 7、8，六项指标：mean advantages、policy gradient loss、KL loss、clip fraction、mean reward、episode success rate）。作者声明 veRL 更新只涉及框架实现、不改算法。

## 六、优点与贡献

1. **新问题**：明确提出并命名 **historical context inconsistency**，说明它是 stepwise group-based RL 的系统性缺陷，而非实现细节。这是最核心贡献。
2. **新分析（诊断）**：用 Oracle 组作偏差参照物，把"上下文不一致 → 优势偏差"从直觉变成可测量量（图 2），并量化 Oracle 步的稀缺性（Table 2）。这套工具比结论本身更易被复用。
3. **新方法**：context-aware hierarchical grouping（式 3–5）把"状态匹配"扩展为"上下文匹配"，用嵌套层级避免样本枯竭；adaptive weighting（式 7）只用单一超参 $\alpha$ 完成多层融合。
4. **新分析（理论）**：Proposition 4.1 给出偏差与方差上下界，证明 HGPO 在步级与 Oracle 之间插值；但依赖较强假设。
5. **工程贡献**：改动集中于优势估计、分组完全离线，因而在**相同显存、相同 rollout、几乎相同时间**下提升，$K=0$ 严格退化为 GiGPO，兼容性与落地成本都好。

## 七、局限与可疑之处

**作者自承的局限：** (1) 层级划分**直接依赖"原始、可切分的历史步"**；若智能体把历史总结进记忆，按步切分就不可行，需改用记忆 embedding 相似度等；(2) 权重是**手工设计的单调族**，作者承认"基于各层优势估计不确定性"的全自适应加权值得探索。

**我认为可疑或没讲清楚的地方：**

1. **偏差单调性假设偏强且未验证**。命题的两条条件是全部结论的基础，论文只用"组越大偏差越小、组越小方差越大"定性辩护，**没有给出任何实测的 $b_k$、$v_k$ 曲线**；图 2 只对比轨迹级/步级/Oracle 三点，中间层是否单调无证据。
2. **$b_k$ 定义不一致的嫌疑**：附录写 $b_k=\mathrm{Bias}[A]=A-A^*$，正文写 $b_k=\mathbb{E}[A^H_k]-A^*$；而界 $b_K\le\mathrm{Bias}[A^H]\le b_0$ 要求 $b_k\ge0$，对归一化后的组相对优势并非必然。有"假设即结论"的循环风险，需对照 PDF 第 18 页核实。
3. **"w Eq. (1)" 消融不可复现**：式 7 的 $w_k$ 只定义在 $k=0..K$ 上，轨迹级优势没有对应下标，论文未说明它用什么权重并入求和。
4. **$K=2$ 时均匀权重（$\alpha=0$）反而更好**，默认却取 $\alpha=1$。自适应加权的价值只在 $K=4$ 显现，而 $K$ 越大高层组越小，收益究竟来自加权还是"多一层平均"并不清楚。
5. **$N=8$ 可能才是真瓶颈**：Oracle 步稀缺本质上是因为每组只有 8 条轨迹，$K=4$ 的高层组常只剩 1–2 个样本。若把 $N$ 提到 16/32，Oracle 组规模与 step utilization ratio 会明显上升，HGPO 相对 GiGPO 的增益可能大幅缩小——**"分层"也许部分只是在弥补 $N$ 太小**。论文完全没有这组实验。
6. **7B 上提升接近噪声**：K=2 的 Out-Suc. 为 92.05±0.22 对 GiGPO 92.18±0.39（**更低**），K=4 WebShop Task Sco. 88.49±0.41 对 89.51±0.59（**更低**）。作者未给逐项显著性检验。
7. **证据面窄**：仅 2 个环境、2 个模型规模，且都是文本域；未与同期 GraphGPO、PA-MoE 对比。
8. **跨版本数值不可比**：GiGPO 与 HGPO 基于更新后的 Verl-agent 重新实现，PPO/RLOO/GRPO 沿用旧结果。Table 7 与 Table 1 差异巨大（1.5B 上 GiGPO K=4 的 In-Suc. 从 85.15 变 93.29），该差异量级**大于 HGPO 相对 GiGPO 的增益量级**。
9. **"只用 Oracle 步训练失败"只有一句话**，无曲线或失败模式分析。
10. **提取文本的排版问题（如实记录，非论文缺陷）**：Proposition 4.1 的两处条件式只剩条件名与后置不等式、前式丢空；式 5 的基数不等式被排成竖排孤立方括号；Appendix B 的 Table 6 中轨迹级公式写成 $\sum_{j\in G_\tau}R(\tau_i)$（应为 $R(\tau_j)$）。这些位置需对照 PDF 核实。

## 八、与同批论文的关系

本批共享**评测环境**的是 **22669_GraphGPO**、**18_PA-MoE**、**13085_AAWM**（都用 ALFWorld 与/或 WebShop）。更关键的是 **GraphGPO 与 HGPO 是同课题组姊妹工作**（共同作者 Xin Cheng、Shuo He、Lang Feng、Bo An，通讯作者同为 Lei Feng），GraphGPO 的参考文献明确引用了 "He et al., 2026"（即本篇），因此总体是**承继 + 互补**：

- **22669_GraphGPO（互补，同一"步级信用分配"赛道的另一条路）**：它把全部 rollout 聚合成状态转移图，用"状态到目标的图距离"给每条边分配图优势，解决**轨迹级归因识别不出"失败轨迹中有价值的步"**；HGPO 解决**步级归组时历史上下文不一致导致的偏差**。两者都以 GiGPO 为主要基线、都用 ALFWorld/WebShop + Qwen2.5-1.5B/7B + group size 8，都在 GiGPO 之上做增量改造——**GraphGPO 改"优势从哪里来"（图距离 vs. 组内奖励统计），HGPO 改"跟谁比"（组怎么组）**。二者原理正交，可组合。
- **18_PA-MoE（互补，改策略网络容量）**：用轻量 phase router 让同阶段路由保持一致，缓解单策略网络的 simplicity bias；同样在 ALFWorld/WebShop 上以 GiGPO 为底座（PA-MoE+GiGPO 报告 93.8% ALFWorld / 82.3% WebShop，Qwen2.5-1.5B）。HGPO 不改网络结构、PA-MoE 不改优势估计，**改动层次不同、可以叠加**。
- **13085_AAWM（互补，改监督信号的目标）**：主张世界模型的训练目标应是"决策所需的环境动力学"而非 next-observation prediction，同样在 ALFWorld/WebShop 评测；与 HGPO 正交，但共享"长时程 + 稀疏奖励"背景。
- **1008_SCA（互补，同属信用分配但作用于动作空间几何）**：指出 group-based 方法只从标量奖励算优势、忽略 GUI 动作（坐标、bbox）的度量结构，用 leave-one-out 局部趋势预测器与 proximity 辅助恢复信号。它按 reward regime 路由组，HGPO 按历史上下文分层——**两者都在抱怨"GRPO 的组内比较对象选得不对"，一个从动作几何切入，一个从上下文一致性切入**，都以 GRPO 为基线。
- **35074_KPO（互补，稳定性视角）**：关注 token 级重要性采样比的方差导致训练崩溃，用因果 Kalman 滤波在线平滑 IS 比，实验主要在数学推理数据。与 HGPO 同属"给 GRPO 家族降偏差/降方差"，但作用对象是 IS ratio（本文式 8 中的 $\rho_\theta$），HGPO 作用在优势 $A^H$ 上，**层次不同、可组合**。
- **相对更远的四篇**：1406_CITA、1423_CIPO、14071_AgentBrew、1432_PhGPO 面向**工具使用/tool planning**环境，不共享评测，也不以 GiGPO 为基线；但都涉及"长时程 + 稀疏最终奖励下的信用分配"——CIPO 用反事实分支 rollout 做粒度选择，CITA 用比较推断模型估下一步工具调用的长时程价值，AgentBrew 用 PMI 把轨迹总信息分解为逐步 credit，PhGPO 用信息素保存可复用转移模式。**共同点是都在为"最终奖励摊到中间步"寻找比组内均值更结构化的替代方案；不同点是 HGPO 是唯一完全不加新模型、不加新 rollout、只重排"比较对象"的解法。**

一句话定位：**HGPO 是这批里"最省"的一篇——把问题定义在分组条件上，用近乎零成本的改动换取优势估计的偏差改进；GraphGPO 与它同门互补，SCA 与它同题（信用分配）不同域，PA-MoE 与它同底座（GiGPO）不同层。**

## 九、如果我要接着做，可以做什么

**方向一：把分层键从"原始状态序列"换成"记忆内容的结构化表示"，正面攻它的自承局限。**
- **改什么**：式 3 的 $\mathcal{C}^k(s_t)$ 要求历史可逐步切分并精确匹配，一旦记忆被摘要压缩就失效。改为"记忆 embedding 相似度分桶"：对每步记忆向量做层次聚类（或按与当前步的余弦相似度分档），得到与式 4 同构的嵌套组 $G^H_k$。
- **为什么可能有效**：摘要记忆（MemAgent、Mem1 一类）恰是长时程任务的主流方向，也正是 HGPO 无法直接套用的场景；推广后能同时拿到"上下文一致"与"记忆可扩展"，且仍不需额外模型。
- **第一步实验**：在 ALFWorld 上做最小改造——把"保留最近 2 步"的记忆换成"一段摘要 + 最近 1 步"，实现两种分组键：(a) 摘要文本 hash 精确匹配，(b) 摘要 embedding 的 $k$-means 分桶。对比 HGPO(原键)/GiGPO/新键 (a)(b)，看 (b) 能否恢复 HGPO 相对 GiGPO 的增益，并按 Table 2 格式报告各层 step utilization ratio。

**方向二：把固定 $\alpha$ 换成自适应权重，并顺手验证 Proposition 4.1 的单调性假设。**
- **改什么**：式 7 现在是全局固定的 $w_k\propto(k+1)^\alpha$，而最优 $\alpha$ 依赖 $K$（$K=2$ 时 $\alpha=0$ 最好、$K=4$ 时 $\alpha=1$）。改成组规模感知权重，或用逆方差加权 $w_k\propto1/(v_k+\lambda)$——正是论文未来工作提到的方向。
- **为什么可能有效**：论文自己说高层组偏差小但样本少、方差大，故最优权重应随 $K$ 与组规模变化；逆方差加权是经典最优线性组合解，理论上直接降低 $\mathrm{Var}[A^H]$。
- **第一步实验**：先做**诊断实验**——按 epoch 记录每层的 $1/|G^H_k|$ 与该层优势相对 Oracle 优势的实测偏差曲线，验证 $b_k$ 递减、$v_k$ 递增（成立即可反驳第七节第 1 点）；再用逆方差权重替换固定 $\alpha$，在 1.5B、$K\in\{2,4\}$ 上与 $\alpha\in\{0,1,2\}$ 对比。

**方向三：与 GraphGPO 正交组合，检验"更好的比较量"与"更好的比较对象"能否叠加。**
- **改什么**：把式 6 组内的 $R(s_n^{(j)})$ 折扣回报换成 GraphGPO 的图距离优势估计，同时保留 $G^H_k$ 层级结构与式 7 加权。
- **为什么可能有效**：两者修的是优势估计的不同环节——**信号来源**（轨迹回报 vs. 图距离）与**参照组**（当前状态匹配 vs. 上下文一致匹配）。若偏差来源可加，组合应同时降低两类偏差；若图距离已隐式引入历史路径信息，则无增益——这也是有价值的负结果。
- **第一步实验**：在**同一代码库**（都用 verl-agent 的 recipe）、同一超参（group size 8、16 groups、160 迭代、3 seeds）下跑 {GRPO, GiGPO} × {原优势, 图优势} 作基线，再加 {HGPO, GraphGPO, 组合}，在 ALFWorld In/Out-Suc. 与 WebShop 两指标上比较；务必**两法都用同一版本框架重训**。

## 十、自测问题

1. 用式 3 的 $\mathcal{C}^k$ 说明：为什么 $G^H_0\supseteq G^H_1\supseteq\cdots\supseteq G^H_K$ 必然成立？这个嵌套性质为 HGPO 带来哪两个具体好处（提示：一个与样本量有关，一个与训练稳定性有关）？
2. 式 7 的 $w_k=(k+1)^\alpha/\sum_k(k+1)^\alpha$ 中，$\alpha=0$ 与 $\alpha\to\infty$ 分别把 $A^H$ 退化成什么估计量？结合 Table 4 解释为什么论文最终选 $\alpha=1$，而不是在 $K=2$ 上表现更好的 $\alpha=0$。
3. HGPO 与 GiGPO 在**算法层面**最本质的差别是什么？给出 $K=0$ 时 HGPO 的形式，并说明消融 "w/o $G^H_1:K$" 实际测的是什么。
4. Proposition 4.1 的结论 $b_K\le\mathrm{Bias}[A^H]\le b_0\le B_T$ 与 $\frac{v_0}{K+1}\le\mathrm{Var}[A^H]\le v_K\le V_T$ 依赖哪两条关于 $b_k$、$v_k$ 的假设？方差下界为什么出现 $K+1$（提示：Cauchy–Schwarz）？若 $b_k$ 的实际单调性不成立，这条命题还剩多少价值？
5. Table 1 中 Qwen2.5-7B 上有两个"HGPO 低于 GiGPO"的格子（$K=2$ 的 ALFWorld Out-Suc.、$K=4$ 的 WebShop Task Sco.）。若让你重新设计这部分实验来支撑"HGPO 更好"，你会增加哪些对照或报告（至少两点）？另外，Table 7 与 Table 1 之间 GiGPO 结果的大幅变化说明了什么实验方法学问题？
