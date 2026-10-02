# PhGPO 精读笔记

论文标题：**PhGPO: Pheromone-Guided Policy Optimization for Long-Horizon Tool Planning**（投稿 NeurIPS 2026，匿名评审版）

本文针对长程工具规划中"探索空间组合爆炸、成功轨迹稀少"的问题：现有 RL 把成功轨迹的价值隐式吸收进策略参数，成功后无法复用，下次还得重新搜索。作者受蚁群优化（ACO）中信息素（pheromone）启发，提出 PhGPO——把已验证成功的工具使用轨迹蒸馏成工具转移图上的显式转移先验（工具-工具边与工具-参数调用边上的信息素），用沉积/蒸发在线维护，并在 rollout 采样时以 log 空间先验引导策略；配套"监督预热 → 渐进式信息素引导 RL → 全信息素 rollout"训练管线。在 Toolathlon、TRAJECT-Bench、TOUCAN 三个长程工具基准、两个 7–8B 骨干上，Match Ratio 与 Task Success Rate 均优于 ReAct、Gorilla、StepTool、TreeRL、GTool 等基线。

## 一、这篇论文的一句话贡献

**把"历史上成功过的工具转移"从隐式的策略参数里拿出来，做成一张带信息素的显式工具转移图，用 ACO 式的沉积与蒸发动它在策略优化期间持续复用，从而在工具空间巨大、成功轨迹稀疏的长程工具规划中同时提升样本效率与最终成功率。**

## 二、读之前需要的前置知识

1. **长程工具使用与组合爆炸**。Episode 是"任务描述 → 调工具 1 → 看返回 → 调工具 2 → … → 完成"的序列。候选工具几百个、步骤十几到二十步时序列数是指数级，且前一步选错会改变后续可达状态。本文引 Toolathlon 说明规模（约 13.48 步/轨迹、618 个基础工具），把"早期选择约束后期状态"当作难度来源。
2. **GRPO 与 group-relative advantage**。对同一 prompt 采 $M$ 条 rollout，用组内相对值当优势 $A_m=(R_m-\bar R)/(\sigma_R+\epsilon)$，无需 value function。它在本文中作用双重：既提供策略梯度更新，又因为学习信号依赖组内各 rollout 的质量差异，一组全是低质量轨迹时优势趋零、没有学习信号——这正是作者论证"信息素引导采样有用"的落点（Sec. 3.3：unguided sampling can yield groups dominated by uniformly low-quality trajectories）。
3. **ACO 的沉积与蒸发**。好路径信息素越积越浓，同时随时间蒸发以防早期偶然路径永久锁死搜索。本文直接搬用 $\tau \leftarrow \mathrm{clip}((1-\rho)\tau+\alpha q_\tau(\xi))$，提供"跨 episode、显式存储、会过期"的经验载体，这是与"把经验塞进网络权重"最本质的区别。
4. **log 空间先验 / 乘积式融合**。把策略分布与先验在 log 空间相加再 softmax，等价于两者相乘后归一化。附录 A 用熵正则化共识问题说明：$\max_q \mathbb{E}_{a\sim q}[\log\pi_\theta+\beta\log\bar\tau]+\eta H(q)$ 的最优解正是 $\tilde\pi\propto\exp(\log\pi_\theta+\beta\log\bar\tau)$，$\eta=1$ 与实现一致。**这是标准结论**，只解释"为何这样融合合理"，不提供收敛性保证。
5. **两层动作空间**。"调哪个工具"由策略 $\pi_\theta(a_t\mid s_t)$ 决定，"参数怎么填"由信息素 $\tau_{\mathrm{Arg}}$ 决定，后者**不参数化、不参与策略优化**。

## 三、研究背景与要解决的问题

作者把前人工作分四类并逐类给出局限：

- **检索增强工具使用**（ReAct、Reflexion）：用检索缩小候选集、取回文档/schema/示例，近期进一步训练 retriever 或与工具调用模型耦合。局限：检索只改善**单步 grounding**，长程轨迹需要**跨多步一致的转移模式**。
- **图结构工具规划与搜索**（ToolNet [28]、NaviAgent [30]、GTool [31]、ControllLLM [29]）：把 API 与依赖组织成图/树做遍历或搜索。局限：这类方法"rely on heuristic or supervised planning signals and **do not maintain an adaptive transition prior distilled from accumulated successful trajectories**"——最直接的靶子。
- **工具使用的强化学习**（StepTool [12] 的过程奖励、ToRL [13] 的结果导向、TreeRL [14] 的树搜索）：局限是成功经验仍**隐式吸收进策略参数**，成功轨迹稀少时学习变慢。
- **经验复用与 agent memory**（Agent KB [38]、AWM [39]、AgentDistill [40]、record & replay [41]）：局限是"rarely convert accumulated successes into **structured transition statistics** that can serve as an explicit transition prior"。

**引用依据**：上述引文之外，长程探索低效引 ARPO [15]、SALT [16]；启发来自蚁群系列 [17,18,42–47] 与把信息素用于推理路径搜索的 [48]——**本文与 [48] 的区别被明确写出：把信息素当作长程工具使用策略优化的显式转移先验，而非独立优化器或类 value 控制器**。

**作者的两阶段预实验（Figure 1）是论点的实证支撑**：先在较易阶段从已验证成功轨迹蒸馏显式跨轨迹转移记忆，再在 GRPO 优化中当先验复用。面板 (a)(b) 显示相同交互预算下平均回报与成功率更高，(c) 发现第一条成功轨迹所需步数更少，(d) 增益随轨迹长度增加（图注 +16%、+55%、+278%）。**这些百分比是预实验图 (d) 的长度分箱标注，不是主实验数字。**

## 四、方法详解（核心）

### 4.1 整体框架与数据流

```
离线数据 D：(任务 x, 参考轨迹 {a*_t}, 执行输出)
   ├─ 工具转移图 G=(V_Tool,E)：v_j 紧接 v_i 出现则加边 (v_i,v_j)
   └─ 每个工具 v_i 的频繁参数调用集 V_Arg(v_i)={v_i^(1..K_i)}，边集 E_arg
阶段 A 监督预热：L_SL = -log π_θ(a*_t|s_t)，不用信息素
阶段 B 渐进式信息素引导 RL（每任务采 M 条轨迹）：
   每步：以概率 p_tf 用参考工具 a*_t；否则 Top-K 候选 → Eq.(7) 融合策略与信息素采样 a_t
         → 由 τ_Arg 采参数调用 u_t → 执行（模拟器）
   组内算 R_m → 优势 A_m → L_mixed 更新 θ
   仅对"验证成功"的轨迹做信息素沉积 (Eq.1) 并写入边记忆库 M_ij
   调度：p_tf↓、λ↓、β↑、w↑、horizon 由短到长
阶段 C 全信息素 rollout：p_tf→0、λ 很小，用 β_max 引导自主 rollout
```

两个必须记住的取舍：**(i) 信息素只由"验证成功"轨迹更新，且只影响 rollout 采样分布，不直接进 loss**（Sec. 3.2：It is not used as a direct step-level reward and does not directly enter the policy optimization loss）；**(ii) 信息素 = 跨任务共享（task-agnostic）+ 按任务检索（task-dependent）加权融合**。

### 4.2 关键公式逐个拆解

**(0) 工具转移图与两层动作。** $V_{\mathrm{Tool}}=\{v_1,\dots,v_N\}$ 为 MCP 暴露的基础工具；$v_j$ 紧跟 $v_i$ 时定义有向边 $(v_i,v_j)\in E$。对每个工具构造频繁参数调用集 $V_{\mathrm{Arg}}(v_i)=\{v_i^{(1)},\dots,v_i^{(K_i)}\}$ 与关联边 $E_{\mathrm{arg}}=\bigcup_{i=1}^{N}\{(v_i,v_i^{(k)})\}$。第 $t$ 步是**分解式决策**：先选 $a_t\in V_{\mathrm{Tool}}$，再选 $u_t\in V_{\mathrm{Arg}}(a_t)$。

**(1) 任务无关信息素的沉积与蒸发（Eq. 1）：**

$$\tau^{\mathrm{agn}} \leftarrow \mathrm{clip}\Big((1-\rho)\tau^{\mathrm{agn}}+\alpha\, q_\tau(\xi),\ [\tau_{\min},\tau_{\max}]\Big)$$

$\rho\in(0,1)$ 是蒸发率，$\alpha>0$ 是沉积率，$q_\tau(\xi)\in[0,1]$ 是该轨迹的沉积分（用与参考轨迹的匹配率度量），$[\tau_{\min},\tau_{\max}]$ 裁剪以防信息素过度集中在极少数转移上。**为什么**：$q_\tau$ 让高质量轨迹贡献更强更新，$(1-\rho)$ 让过期统计自然失权，裁剪保证 $\log\tau$ 有界。$\tau^{\mathrm{agn}}_{\mathrm{Tool}}$ 与 $\tau^{\mathrm{agn}}_{\mathrm{Arg}}$ 同式维护。

**(2) 任务相关信息素的检索（Eq. 2）与估计（Eq. 3）。** 每条边维护记忆库 $M_{ij}$ 存 $(e,q_\tau)$：$e=\phi(x)$ 是任务嵌入（句编码器用 `all-MiniLM-L6-v2`），$q_\tau$ 是包含该转移的某条已验证成功轨迹的沉积分。给定任务 $x$，取相似度超阈值的记忆并按相似度加权平均：

$$M_{ij}(x)=\{(e,q_\tau)\in M_{ij}\mid \mathrm{sim}(e_x,e)\ge\theta_{\mathrm{sim}},\ \forall e\}$$

$$\tau^{\mathrm{dep}}(v_i,v_j\mid x)=\tau_0+\frac{\sum_{(e,q_\tau)\in M_{ij}(x)}\mathrm{sim}(e_x,e)\,q_\tau}{\sum_{(e,q_\tau)\in M_{ij}(x)}\mathrm{sim}(e_x,e)+\epsilon_{\mathrm{dep}}}\cdot(\tau_{\max}-\tau_0)$$

$\tau_0$ 是基础值，$\epsilon_{\mathrm{dep}}$ 为数值稳定小常数，$(\tau_{\max}-\tau_0)$ 把结果缩放到与任务无关信息素可比的量级。**作用**：当前任务语义上像某条历史成功轨迹时，那条轨迹用过的转移更该被选（纯全局信息素对具体任务可能文不对题）。

**(3) 置信度与融合（Eq. 4、Eq. 5）：**

$$c_{ij}(x)=\min(1,n_{\mathrm{act}}/n_{\min})\cdot s_{\max}\cdot \bar q_\tau$$

$$\tau_{\mathrm{Tool}}(v_i,v_j\mid x)=\big(1-w\,c_{ij}(x)\big)\,\tau^{\mathrm{agn}}_{\mathrm{Tool}}(v_i,v_j)+w\,c_{ij}(x)\,\tau^{\mathrm{dep}}_{\mathrm{Tool}}(v_i,v_j\mid x)$$

$n_{\mathrm{act}}$ 为命中记忆数，$n_{\min}$ 为预设最小条目数，$s_{\max}$ 为最大相似度，$\bar q_\tau$ 为命中记忆平均沉积分，$w$ 控制任务相关项最大贡献。**设计要点**：若某条边只检索到很少记忆、或语义相似但历史质量一直很差，置信度就低，融合结果自动退回任务无关信息素（附录 B.4 称保持先验"conservative"）。

**(4) 监督预热（Eq. 6）：** $\mathcal{L}_{\mathrm{SL}}=-\log \pi_\theta(a_t^{*}\mid s_t)$，$a_t^{*}$ 为参考下一步工具；此阶段**不用信息素引导**，目的是把策略从"随机选工具"推到"会选工具"并提供稳定初始化。

**(5) 信息素引导采样（Eq. 7）——本文最核心的一步：**

$$p(a_t\mid s_t,a_{t-1})\propto \exp\Big(\log \pi_\theta(a_t\mid s_t)+\beta\log \tau_{\mathrm{Tool}}(a_{t-1},a_t\mid x)\Big)$$

先在 $\pi_\theta$ 下取 Top-$K$ 候选作为采样支撑集再按上式采样；$\beta\ge0$ 控制先验强度（$K=20$，$\beta$ 线性调度到 $\beta_{\max}=0.7$）。**为什么这样设计**：log 空间相加 = 分布相乘，等价于要求被选动作既被神经策略支持、又被累积转移统计支持；条件用 $a_{t-1}$ 正好体现"转移"这一层结构。选定 $a_t$ 后 $u_t$ 按 $\tau_{\mathrm{Arg}}(a_t,u_t\mid x)$ 选，不参数化。

**(6)–(7) 优势与策略目标（Eq. 8、Eq. 24）：** $A_m=\dfrac{R_m-\bar R}{\sigma_R+\epsilon_{\mathrm{adv}}}$，$\bar R=\frac1M\sum_m R_m$（$M=5$）；$\mathcal{L}_{\mathrm{PG}}=-\mathbb{E}\big[\min(r_t(\theta)A_m,\ \mathrm{clip}(r_t(\theta),1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}})A_m)\big]$，$r_t(\theta)={\pi_\theta(a_t\mid s_t)}/{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}$。（正文中该式与 Eq. (8) 排版粘连、无独立编号；附录 B.3 重写为 Eq. (24) 并带上 rollout 下标 $m$。）

**(8) 混合行为与混合目标（Eq. 9、Eq. 10）：**

$$a_t=\begin{cases}a_t^{*}, & \text{以概率 } p_{tf}\\ \text{由 Eq. (7) 采样}, & \text{以概率 } 1-p_{tf}\end{cases}\qquad \mathcal{L}_{\mathrm{mixed}}=\lambda\mathcal{L}_{\mathrm{SL}}+(1-\lambda)\mathcal{L}_{\mathrm{PG}}-\gamma H[\pi_\theta]$$

附录 B.3 写成 $\pi_{\mathrm{mix}}=p_{tf}\cdot\delta(a_t=a_t^{*})+(1-p_{tf})\cdot\tilde\pi(a_t\mid s_t)$，并特别说明**即使 oracle 选出的动作，也仍按其在策略下的概率计算比率**，因为它是 $\pi_{\mathrm{mix}}$ 下真正被执行的动作。$p_{tf}$ 衰减、$\lambda$ 向下退火、horizon 课程由短到长、$\beta$ 与 $w$ 随累积轨迹增加而上调。

**(9) 全信息素 rollout（Eq. 11）：** $\tilde\pi(a\mid s_t)\propto \exp\big(\log \pi_\theta(a\mid s_t)+\beta_{\max}\log \tau_{a_{t-1},a}\big)$，$\tau_{a_{t-1},a}$ 是融合后工具转移信息素的简写。

**(10) 附录支撑公式（别当新结果）**：Eq. (13)–(22) 是熵正则化共识问题的推导与 KL 投影视角；Eq. (25)(26) 是 $r_t=r_t^{\mathrm{intent}}+r_t^{\mathrm{exec}}$ 与 $R=\sum_{t=1}^{T}r_t+r^{\mathrm{outcome}}$（intent：精确匹配参考工具 +0.5、同类目 +0.2、从上一步错误恢复 +0.1；execution：模拟器有效执行 +0.5、报错或超时 −0.5）；Eq. (27) 是沉积分 $q_\tau(\xi)=\mathrm{Match}(\xi,\xi^{*})=\frac1T\sum_{t=1}^{T}\mathbb{I}[\mathrm{tool}(a_t)=\mathrm{tool}(a_t^{*})]$；Eq. (28) 是只在非 teacher-forced 步上算的 $q_{\mathrm{model}}$；Eq. (31) 是多样性 $H(\pi)/\log|A|$；Eq. (32)–(35) 是 GRPO/PPO/RLOO 三种可选骨干。

### 4.3 训练流程（Algorithm 1）

① 输入数据集 $\mathcal{D}$、horizon 课程 $\{L_s\}$、四条调度 $p_{tf}(s),\lambda(s),\beta(s),w(s)$、组大小 $M$、工具图与调用边、句编码器 $\phi$、ACO 参数 $(\rho,\alpha,\tau_{\min},\tau_{\max},\tau_0)$、$\theta_{\mathrm{sim}},n_{\min},\epsilon_{\mathrm{dep}}$。② 初始化 $\tau^{\mathrm{agn}}\leftarrow\tau_0$、记忆库为空。③ 子过程 ComputeFusedPheromone：检索 $M(x)$ → 算 $\tau^{\mathrm{dep}}$ 与 $c(x)$ → 融合并裁剪。④ 阶段一若干 epoch 最小化 $\mathcal{L}_{\mathrm{SL}}$。⑤ 阶段二（$s=1..S$）：设定 $L_s$ 与四条调度；对每个 $(x,\{a_t^{*}\})$ 算 $e_x=\phi(x)$，采样 $M$ 条轨迹，每条走 $L$ 步——构造状态 $s_t$（任务描述 + 定长截断的近期工具交互历史），以概率 $p_{tf}$ 取 $a_t=a_t^{*}$，否则对 Top-$K$ 候选逐个算融合信息素并按 Eq. (7) 采样，再由 $V_{\mathrm{Arg}}(a_t)$ 采 $u_t$，执行并追加；结束后算 $R_m$ 与 $q_\tau(\xi_m)$。⑥ Eq. (8) 算优势 → Eq. (10) 更新 $\pi_\theta$。⑦ 对每条**验证成功**的轨迹，遍历工具转移边用 Eq. (1) 更新 $\tau^{\mathrm{agn}}_{\mathrm{Tool}}$ 并存入 $M_{ij}$，调用边同理存入 $M_{i,u}$。

### 4.4 相较基线方法的改动点具体在哪

以 GRPO 为对照，改动精确落在四处：**① rollout 采样分布被替换**（从 $\pi_\theta$ 换成 Eq. (7) 的 $\exp(\log\pi_\theta+\beta\log\tau)$）——唯一直接改变"数据从哪来"的改动，也是论文主张的主要增益来源；**② 新增跨 episode 的显式状态**（图上信息素 + 边记忆库，靠 Eq. (1) 维护）——相对 GRPO "经验全在权重里"的唯一结构性新增；**③ 新增参数调用决策通道**（$u_t$ 由 $\tau_{\mathrm{Arg}}$ 决定，不参数化、不训练）；**④ 新增训练调度**（$p_{tf}$ 衰减 + $\lambda$ 退火 + horizon 课程 + $\beta,w$ 上调；消融显示去掉 Mixed Curriculum 掉分最多，说明它不是装饰）。**策略梯度损失本身没有改**——附录 H 还专门证明换成 PPO / RLOO 也成立。

## 五、实验怎么做的

**数据集（Table 3，均按两层动作空间重新统计）**：Toolathlon（MCP apps，618 个 L1 工具、1,250 个 L2 参数调用模式、108 tasks、平均 13.48 步）；TRAJECT-Bench（executable APIs，381 / 715 / 5,870 queries / 6.36 步）；TOUCAN 仅 Multi-step 子集（MCP servers，850 / 15,294 / 1,646,546 traj. / 3.16 步）。每条实例转成含"任务描述 + 参考轨迹 + 执行输出"的 episode，只保留参考轨迹可验证成功的 episode，固定种子划分 train/val/test。

**基线（4 组）**：Prompting/Planning（ReAct、Plan-and-Solve、Beyond-ReAct）；Retrieval（Gorilla、ToolReAGt、Tool Graph Retriever）；RL（TreeRL、ToRL、StepTool）；Graph-based（GAP、GTool、NaviAgent、ToolNet）。骨干 Qwen2.5-7B 与 Llama3.1-8B，**所有基线在同一动作空间与执行接口下运行**。指标为 Match Ratio（序列级，衡量生成轨迹与参考轨迹在工具身份与顺序上的相似度）与 TSR（执行级，由 **GPT-4o 作为 outcome judge** 在固定协议下判定，只看结果、不要求顺序一致）。

**实现细节**：LoRA（$r=64$、$\alpha=128$）；temperature 0.7、top-$K$ 采样 $K=20$、随机探索 $p_{\mathrm{rand}}=0.05$；$\beta$ 线性调度到 $\beta_{\max}=0.7$；最大轨迹长度 20；$M=5$。

**主要结果（Table 1，单元格式 Match R. / TSR）**：

| 方法 | Toolathlon Qwen | Toolathlon Llama | TRAJECT Qwen | TRAJECT Llama | TOUCAN Qwen | TOUCAN Llama |
|---|---|---|---|---|---|---|
| ReAct（提示类代表） | 5.31 / 2.92 | 5.15 / 2.48 | 15.32 / 9.20 | 9.16 / 7.01 | 11.64 / 8.92 | 7.03 / 5.88 |
| Gorilla（检索类最强） | 6.60 / 3.26 | 5.78 / 3.65 | 29.04 / 22.62 | 19.38 / 11.42 | 22.32 / 16.05 | 18.99 / 16.94 |
| StepTool（过程奖励 RL） | 14.33 / 12.08 | 11.58 / 7.87 | 35.22 / 23.18 | 26.31 / 20.12 | 19.75 / 20.86 | 14.46 / 15.30 |
| TreeRL（树搜索 RL） | 21.26 / 18.91 | 18.78 / 11.42 | 35.45 / 27.82 | 26.33 / 20.04 | 30.65 / 18.74 | 20.20 / 18.62 |
| GTool（图导航，最强基线） | 20.60 / 17.82 | 16.50 / 14.22 | 32.50 / 28.06 | 26.20 / 21.78 | 27.56 / 19.35 | 19.50 / 13.33 |
| **PhGPO（Ours）** | **25.25 / 20.34** | **22.83 / 19.76** | **41.62 / 32.18** | **31.84 / 22.66** | **35.17 / 26.28** | **24.67 / 19.86** |

原表还有 Plan-and-Solve、Beyond-ReAct、ToolReAGt、Tool Graph Retriever、ToRL、ToolNet、GAP、NaviAgent 共 8 行基线（数值未抄录），**它们在任何基准上都没有超过 PhGPO**。**数字在提取文本中按列纵向排列，逐格核对需对照 PDF 表 1**。

**消融（Table 2，Qwen2.5-7B，Toolathlon）**，以 Full 的 25.25 / 20.34 为基准（Match R. 与跌幅 / TSR 与跌幅）：w/o Pheromone（全程 $\beta=0$）21.23（−4.02）/ 16.72（−3.62）；**w/o Mixed Curriculum 19.34（−5.91）/ 14.86（−5.48），掉分最大**；Static Prior 23.93（−1.32）/ 18.71（−1.63）；w/o Evaporation（$\rho=0$）21.72（−3.53）/ 16.28（−4.06）；w/o Task-dependent Pheromones 22.45（−2.80）/ 17.35（−2.99）。

**其余实验**：**$\beta$ 消融（Table 4）** 动态 $\beta$ 25.25 / TAcc 27.16 / 多样性 0.58，优于 $\beta=0$（21.23 / 19.86 / 0.76，多样性过高=无焦点探索）、$\beta=1$（23.93 / 22.05 / 0.53）与 $\beta=5$（17.80 / 16.50 / 0.25，多样性过低=困在局部最优）。**优化器无关性（Table 6，Qwen）** GRPO 25.25→21.23（去引导）、PPO 23.27→20.78、RLOO 20.21→18.61。**随机探索 vs 信息素（Table 8）** 去随机探索只掉到 24.63 / 26.34 / 0.52，$p_{\mathrm{rand}}$ 升到 0.10 也只到 24.38 / 26.47 / 0.66，说明**信息素才是定向探索的主要来源**。**开销（Table 7）** 每训练步 11.89s vs GRPO 11.86s，每推理步 59.3ms vs 58.5ms，峰值显存 17.4GB vs 17.3GB。**工具演化（附录 L）** 把常用原子工具链合并成 macro tool 模拟 API 升级，局部图/记忆更新用约 16% 的原始 RL 步数恢复到改动前约 87% 的性能，全局重初始化需约 38% 步数。**路径发现（附录 J）** 以 $q_\tau(\xi)\ge0.6$ 定义高质量轨迹，PhGPO 比 $\beta=0$ 更早覆盖（图注 1173 vs 2500 steps），长度分箱加速 +25%/+21%/+20%/+21%/+23%。**可视化（Figure 3、7）** 信息素矩阵随训练越来越集中在参考链上（如 WebSearch→FetchHTML→WriteFile→ReadFile→RunCmd），非参考转移在蒸发下淡出。

**执行环境的说明（附录 B.6、K）**：训练**不调用真实外部 API**，而走工具输出模拟器——可回放离线缓存输出，也支持可插拔 LLM 模拟器后端（实际用 `qwen3-235b-a22b-thinking-2507`），用 `«END»` 标记做完成检测、用内容哈希缓存输出；只有 TSR 由 GPT-4o 判定。

## 六、优点与贡献

1. **重新刻画了一个被忽视的失效模式（新问题视角）**：成功轨迹只贡献即时奖励、其转移模式隐式沉入参数、成功后仍需重新搜索。Figure 1 用受控两阶段预实验检验它（同一 GRPO、同一交互预算，只加/不加显式先验）。
2. **机制清晰且可插拔（新方法）**：显式先验被放在"rollout 采样"这一层而非 reward 或 value，因此天然与任何 group-based 优化器兼容（附录 H 用 GRPO/PPO/RLOO 支持）。
3. **两级信息素 + 置信度门控有细节**：任务无关（全局统计）与任务相关（按任务嵌入检索的边记忆）融合，并用 $c_{ij}$ 只在证据相关且可靠时提高任务相关项权重，否则退回全局先验。
4. **渐进课程被消融证明必要（新分析）**：$p_{tf}$ 衰减 + $\lambda$ 退火 + horizon 课程 + $\beta/w$ 上调，让信息素先积累再转向自主 rollout；Table 2 显示去掉它是所有消融里掉分最大的。
5. **工程性分析与可视化较完整**：开销可忽略、工具演化下局部更新的恢复效率、信息素在图上的可见收敛、以及诊断用的 $q_{\mathrm{model}}$（只在非 teacher-forced 步上算匹配率）。

## 七、局限与可疑之处

**作者自承的局限**：Sec. 5 的"Limitations"实际只写未来方向——更广的验证设定、更丰富的参数调用建模、API 不断演化的动态工具环境；附录 L/M 补了工具更新与开销讨论。

**我认为可疑或没讲清楚的地方：**

1. **信息素更新依赖"参考轨迹"这一强假设。** Eq. (27) 的沉积分是**按步对齐的工具身份匹配率**，即"质量"由"像不像参考路径"定义，而非"任务是否完成"；但附录 D 又承认"a task can sometimes be completed through a valid alternative tool-use path even when its Match Ratio to the reference trajectory is not perfect"。**一条成功但顺序不同的轨迹，$q_\tau$ 可能很低、其转移模式得不到强化**——本文可能把"参考路径对齐"误当成"可复用经验"的代理。
2. **数学推导有被过度包装的风险。** 附录 A 把 Eq. (7) 说成"熵正则化共识问题的闭式解"，但这只是标准结论（带熵正则的期望对数收益最大化 → Gibbs/softmax，$\eta=1$ 退化），**不提供收敛性或"信息素先验为何能加速学习"的任何保证**；checklist 第 3 项也承认论文没有形式化定理与证明。
3. **度量链条很长，LLM 承担两处关键角色。** 训练时的执行反馈来自 `qwen3-235b-a22b-thinking-2507` 模拟器（带缓存与 pool-based fallback），测试时的 TSR 由 GPT-4o judge。论文未报告模拟器输出与真实工具返回的一致性校验、judge 与人工标注的一致率、或换 judge 时 TSR 的稳定性。
4. **实验规模与统计强度不足。** Toolathlon 只有 108 tasks，消融全在 Qwen2.5-7B 的**单数据集、单次运行**上报告；主表无误差棒或多种子（唯一带 ±1 标准差的是附录 G 超参敏感性，3 seeds）。而 Static Prior 与 Full 的差只有 1.32 / 1.63 个百分点，**没有方差信息就无法判断是真实效应还是噪声**。
5. **基线公平性存在一个未量化的偏置。** 附录 B.6 对需要工具格式感知的基线统一做了"same lightweight tool-format fine-tuning"，这消除了格式差异干扰，但论文没给"不做这一步时基线掉多少"的对照，无法判断优势有多少来自算法本体。
6. **"长程"表述与 TOUCAN 实际长度不匹配。** 三个基准被统称 long-horizon，但 TOUCAN 平均只有 3.16 步（TRAJECT-Bench 6.36），只有 Toolathlon 是 13.48；"长程"在这里承担了两种不同难度来源。
7. **参数调用通道不对称。** $u_t$ 完全由 $\tau_{\mathrm{Arg}}$ 决定、不受策略学习，也没有"把参数调用交给策略"的消融；作者把"学会怎么调用"列为贡献之一（Sec. 1），但它是被检索出来的，不是被学出来的。
8. **更简单的替代解释未被排除。** 信息素引导采样本质上是在 Top-$K$ 候选上做一次**有偏、跨 episode、基于历史成功统计的重加权**，类似"无梯度的频率先验"或粗略的经验回放偏置。论文未与"用同样成功轨迹做 SFT/拒绝采样微调"或"朴素频率先验"对照，因此"必须用 ACO 沉积-蒸发形式"未被证明——Table 4 中固定 $\beta=1$ 也有 23.93（vs 21.23），只比动态调度低 1.32。
9. **提取文本层面**：Figure 2、3、7 的图内文字被打散成单字符竖排，图注完整但图内标签只能部分还原；正文中策略梯度损失式与 Eq. (8) 排版粘连、无独立编号，需对照 PDF 第 6 页核实。

## 八、与同批论文的关系

本批按问题域可分三类，PhGPO 属"长程工具使用 agent 的 RL 训练"这一簇。

- **`1406_CITA`（最直接的竞争者/互补者）**：CITA 用配对信号训练比较推断模型（CIM）估计下一次工具调用的长程价值再引导 GRPO，**评测环境完全重合**（Toolathlon/TOUCAN/TRAJECT-Bench，骨干同为 Qwen2.5-7B 与 Llama3.1-8B）。差别在引导信号载体：CITA 是**参数化价值估计器**，PhGPO 是**非参数、显式存于图上的转移统计**；两条路线可互为基线，也可互补。
- **`1423_CIPO`（共享评测环境的另一竞争者）**：CIPO 做"原子工具 vs 复合 skill"的粒度选择（skill 压缩轨迹长度），PhGPO 改采样分布的引导来源，**改动点正交**；skill 压缩会让信息素图更稀疏、蒸发更快，值得研究二者交互。
- **`14071_AgentBrew`（互补，设定对立）**：AgentBrew 前提是"无预定义任务、无 verifier、无可靠模拟器"，只能从**原始未筛选**轨迹离线学习（retrospective task inference + PMI credit）；**PhGPO 前提恰好相反**（需可验证成功 episode、需模拟器反馈、靠 on-policy rollout），而 AgentBrew 的 PMI credit 可替换 PhGPO 里"按参考匹配率"定义的 $q_\tau$。
- **`15751_HGPO`（承继/互补）**：HGPO 处理 step-wise group-based RL 的 context inconsistency（同组内各步历史上下文不同导致优势有偏）。**PhGPO 引用了同一作者群的 Group-in-Group Policy Optimization（ref [50]）并以 GRPO 为骨架**，因此 HGPO 是其优化骨架的潜在升级件，可缓解"信息素只管采样、信用分配仍粗粒度"的问题。
- **`22669_GraphGPO`（概念近亲，机制上竞争）**：GraphGPO 把 rollout 聚合成统一 state-transition graph，用图上到目标的距离给每条边分配图优势——**本批唯一与 PhGPO 一样显式建图的工作**，但图的作用不同：GraphGPO 用于**信用分配**，PhGPO 用于存**采样先验**；两者可合成一张图。
- **`1008_SCA`（同族，模态不同）**：SCA 指出 GRPO 的标量优势在 GUI 动作空间丢掉几何方向信息、all-miss group 无学习信号，与 PhGPO **共享"GRPO 在特定结构下信号不足"的诊断**（PhGPO 说的是大工具空间下组内全是低质量轨迹），但模态不同、无共享基准。
- **`35074_KPO`（正交互补的底层工具）**：KPO 用 Kalman 滤波平滑 token-level importance sampling 比值以防训练崩溃，**正好补 PhGPO 的薄弱点**——PhGPO 的 rollout 来自 $\pi_{\mathrm{mix}}$ 与 $\tilde\pi$，却只记录 $\pi_{\theta_{\mathrm{old}}}$ 下的 log 概率，off-policy 程度高于标准 GRPO。
- **`18_PA-MoE` / `13085_AAWM`（思路互补，设定无关）**：PA-MoE 的 phase-aware 路由与 PhGPO 的"按训练阶段切换机制"同源；AAWM 则与 PhGPO **共享"转移证据要显式化"的诊断**，只是一个走"图上的信息素 + 边记忆库"、一个走"决策需求驱动的跨轨迹证据合成"。

**小结**：直接竞争者是 **CITA（1406）与 CIPO（1423）**；概念上最近的兄弟是 **GraphGPO（22669）**；最互补的是 **AgentBrew（14071）**。

## 九、如果我要接着做，可以做什么

**方向 1：把 $q_\tau$ 从"像不像参考路径"换成"任务是否真的成功"。**
- 要改什么：Eq. (27) 的沉积分改为不使用参考轨迹的信号——(a) verifier 对任务完成的判定；(b) `14071_AgentBrew` 的 PMI credit，把轨迹对最终指令的信息量分解到每条转移边；(c) `1406_CITA` 的 CIM 输出。
- 为什么可能有效：附录 D 自己承认存在合法但顺序不同的成功路径；结果导向的 $q_\tau$ 能把这类路径的转移模式也沉积下来，避免信息素只集中在"恰好是参考顺序"的边上。
- 第一步实验：在 TRAJECT-Bench 上固定其他一切、只替换 $q_\tau$ 定义，比较三种定义下的 Match Ratio 与 TSR。若 TSR 升而 Match Ratio 不降，说明结果导向的 $q_\tau$ 捕捉到了参考匹配率漏掉的可复用经验。

**方向 2：把 GraphGPO 的图距离场接到 PhGPO 的同一张转移图上，做"采样先验 + 信用分配"双用途图。**
- 要改什么：PhGPO 的信息素只影响 rollout 采样，优势仍是 trajectory-level 的 $A_m$；而 `22669_GraphGPO` 已证明用图到目标的距离可给出 edge-level 图优势。让两者共用同一 $V_{\mathrm{Tool}}$ 与 $E$，在 $\mathcal{L}_{\mathrm{PG}}$ 里用图优势替换或混合 $A_m$；可参考 `15751_HGPO` 的层次化 step-level 优势以避免 context inconsistency。
- 为什么可能有效：PhGPO 最"粗"的一环就是信用分配——13 步轨迹只给一个标量回报，中间步骤贡献无法区分，而这正是 GraphGPO 解决的问题；共用一张图时边际成本几乎为零。
- 第一步实验：在 Toolathlon 上做三方对照——(i) 原始 PhGPO、(ii) PhGPO + GraphGPO 图优势、(iii) 纯 GraphGPO；重点统计失败轨迹中"使状态更接近目标"的边是否获得了正优势。

**方向 3：修正 off-policy 比值的不一致，并检验"ACO 形式是否必要"。**
- 要改什么：真实行为分布是 $\pi_{\mathrm{mix}}=p_{tf}\delta(a_t^{*})+(1-p_{tf})\tilde\pi$，但 $r_{m,t}=\pi_\theta/\pi_{\theta_{\mathrm{old}}}$ 用的是**未加信息素、也未含 oracle 混合**的策略概率，可把分母换成在 $\pi_{\mathrm{mix}}$（或 $\tilde\pi$）下计算的概率。同时加一组"朴素频率先验"对照：把 $\tau$ 换成历史成功轨迹上该转移的原始出现次数（不蒸发、不裁剪）。
- 为什么可能有效：分母换成真实行为分布后比值更接近 1、梯度方差应下降——正是 `35074_KPO` 处理的问题。而"朴素频率先验"对照能回答更基础的问题：收益来自"显式历史统计"这一想法，还是来自 ACO 沉积-蒸发这一具体形式？
- 第一步实验：在 Toolathlon + Qwen2.5-7B 上跑四组（默认 / 分母改用 $\pi_{\mathrm{mix}}$ / 前者 + KPO 滤波 / 朴素频率先验），记录训练是否崩溃、IS 比值方差、最终 Match Ratio 与 TSR。若"朴素频率先验"与 PhGPO 差距落在噪声范围内，则 Table 4 中固定 $\beta=1$（23.93 vs 25.25）已暗示这一结论。

## 十、自测问题

1. **（方法内部细节）** 采样分布 Eq. (7) 是 $\exp(\log\pi_\theta+\beta\log\tau_{\mathrm{Tool}})$，而策略梯度损失里的比值分母是 $\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$。**这两者是否自洽？** 如果行为分布实际上由 $\pi_{\mathrm{mix}}$ 生成，Eq. (24) 的 importance ratio 在什么意义上仍有效？你会在哪里修改它，修改后可能带来什么后果（提示：比较 oracle 步骤与非 oracle 步骤的比值分布）。
2. **（方法内部细节）** 解释置信度 $c_{ij}(x)=\min(1,n_{\mathrm{act}}/n_{\min})\cdot s_{\max}\cdot\bar q_\tau$ 中三个因子各自的含义，并说明：**为什么作者要把平均沉积分 $\bar q_\tau$ 也乘进置信度，而不只用检索到的记忆数量？** 如果去掉 $\bar q_\tau$，什么情况下融合后的 $\tau$ 会变得有害？
3. **（方法内部细节）** 信息素只由"验证成功"轨迹更新，沉积分用的是按步匹配率 Eq. (27)。**假设某任务存在一条合法但顺序完全不同的成功轨迹，它的 $q_\tau$ 会怎样？这会对该任务后续的信息素分布产生什么系统性影响？** 附录 D 的哪句话实际上承认了这种现象的存在，却没有在机制上处理它？
4. **（实验设计）** Table 2 的消融只报告 Toolathlon 上 Qwen2.5-7B 的**单数据集、单次运行**结果，而 Static Prior 与 Full 的差距只有 1.32（Match R.）/ 1.63（TSR）。**请设计一个能判断该差距是否显著的最小实验方案**：需要多少种子、报告什么统计量、在哪些数据集上重复？并说明为什么"固定测试划分 + 确定性指标"（checklist 第 7 项的回答）不足以回答这个问题。
5. **（实验设计 + 批判性）** TSR 由 GPT-4o judge 判定，训练时的执行反馈由 `qwen3-235b-a22b-thinking-2507` 模拟器提供，而沉积分 $q_\tau$ 来自参考轨迹匹配率。**请指出这三个信号源之间可能产生的不一致，并说明它如何影响结论可靠性**；再提出至少一个具体诊断实验（例如换 judge 模型重测 TSR，或统计"高 TSR 但低 Match Ratio"的实例比例），并说明你预期看到什么结果才算论文结论成立。
