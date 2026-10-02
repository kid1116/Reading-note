# 精读笔记：Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making

> 论文标识：13085_Beyond_Next_Observation_ ｜ 专题：主题三 世界模型与序列决策 ｜ 短名：AAWM
> 匿名 ACL 投稿（标题下写 "Anonymous ACL submission"）。

现有 LLM 智能体的世界模型训练几乎都把目标设成"预测下一观测"，监督内容因此由环境恰好暴露了什么决定，而非由策略决策时真正需要什么决定。作者提出 **AAWM**：在每个决策上下文先让策略自陈"已确信的模式"与"待解决的疑问"（Self-Probing），用这些命题到全局迁移池检索证据（Transition Retrieval），再由外部模型综合成自然语言世界模型目标（Dynamics Synthesis），据此微调策略。在 ALFWorld、WebShop 与 AgentGym 四环境、两个模型规模、SFT 与 GRPO 两种设置下一致优于下一观测预测基线，最高提升 **6.3** 和 **6.2** 个成功率点。

## 一、这篇论文的一句话贡献

**把"世界模型的训练目标该写什么"重新定义为一个 target selection（目标选择）问题，并给出由策略自身信念与疑问决定监督内容的数据构造流程 AAWM——不改模型结构、不改 RL 算法，只改 SFT 目标内容，就稳定超过 next-observation prediction。**

## 二、读之前需要的前置知识

1. **部分可观测文本环境与决策上下文 $o_t$**：智能体只看到环境返回的文字，输出动作，环境返回下一段文字；这一步输入记作 **$o_t$**。关键是同一段 $o_t$ 对"重建观测"与"做决策"价值不同——重建要覆盖全部文字，决策只要其中能改变动作选择的部分。
2. **World model 与 next-observation prediction**：在 LLM 智能体里，"世界模型"通常不是独立网络，而是**给策略加的辅助训练目标**：输入 $(o_t,a_t)$、输出下一观测或其摘要。本文基线 **IWM（Implicit World Modeling，Zhang et al., 2025）** 即此范式。
3. **SFT / imitation learning / GRPO**：管线三段——① world modeling SFT；② 轻量 imitation learning（少量轨迹上学直接输出动作）；③ GRPO（同一 prompt 采样一组回答，用组内相对奖励做策略梯度）。**AAWM 与 IWM 的差别只在第 ① 步的目标内容**。
4. **MMR（最大边际相关性）**：选下一条检索结果时既要"和查询像"又要"和已选的不像"，用它让 $K$ 条证据覆盖被查询动力学的不同侧面。
5. **LLM-as-judge**：用更强模型对生成结果打 0/1 分。**注意本文裁判与数据构造用的是同一个模型**（见第七节）。

## 三、研究背景与要解决的问题

**前人怎么做。** 两支：一是 **Prompting 类**（CoT、ReAct、Reflexion、memory/retrieval/tool-use），只改善**推理时**怎么选动作，"do not change the supervision used to train the policy"；二是**世界模型类**，作者列举 Chae et al. 2025（transition 级状态变化摘要）、Berg et al. 2025（语义压缩）、Guo et al. 2025（结果状态预测）、Yu et al. 2025（动作模拟）、Qiao et al. 2024（知识注入）、Chen et al. 2025（从自身 rollout 内化动力学）、Wang et al. 2026（强化状态预测）、Hu et al. 2025（Agent2World）、Zhang et al. 2025（自造交互数据）。它们**只改了目标的形式**（更紧凑、更语义、更 planning-friendly），没改来源。

**具体局限。** 目标内容"is still determined by the transition exposed in the dataset"，因此仍是环境驱动而非决策驱动——**再漂亮的摘要，也还是环境那一格"恰好翻出来的牌"。**

**论证该局限的两条引用。** Nair et al. (2020) *Goal-aware prediction*：策略需要的是影响下一步动作的状态信息，而非对下一观测的完整重建；Li et al. (2024) *Predicting vs. acting*：优化模型做目标导向行为会**降低**预测精度，说明预测质量与决策质量不是同一目标。问题由此被重述为 **core target selection problem**："what environment dynamics does the policy need in the world modeling target before acting?" **这属于"新问题"层面的贡献。**

## 四、方法详解（核心）

### 4.1 整体框架与数据流

环境 $\mathcal{E}$ 部分可观测，策略 $\pi_\theta$ 在步 $t$ 收到 $o_t$，采样 $a_t\sim\pi_\theta(\cdot\mid o_t)$，环境返回 $o_{t+1}$；轨迹集 $\mathcal{D}=\{\tau_i\}$，$\tau_i=(o_0,a_0,o_1,\dots,o_T)$。所有一步迁移记录为

$$U(\mathcal{D})=\{(o_t,a_t,o_{t+1})\}$$

两个集合：**全局迁移池 $\mathcal{T}$**（所有轨迹记录汇成可检索集合）；**即时迁移集 $\mathcal{I}_t\subseteq\mathcal{T}$**（来自**同一上下文 $o_t$** 的记录，含日志那一条，以及在同状态额外执行其他动作得到的迁移）。

策略先"提问"，检索去找"证据"，外部模型写"答案"，答案就是训练目标。

### 4.2 关键公式逐个解释

**（基线）下一观测预测目标**，式 (1)：

$$\mathcal{L}_{\text{obs}}(\theta)=-\sum_{(o_t,a_t,o_{t+1})\in U(\mathcal{D})}\log p_\theta(\bar{o}_{t+1}\mid o_t,a_t) \tag{1}$$

$\bar{o}_{t+1}=g(o_{t+1})$ 是由下一观测构造的文本目标，$g$ 可以是恒等映射、状态变化抽取器或摘要器；损失即"给定 $(o_t,a_t)$ 生成 $\bar o_{t+1}$"的负对数似然。**它在做什么**：把监督绑死在"环境这次返回了什么"；但决策所需的动力学可能稀疏、隐含，甚至散落在更早的迁移里。

**（第一步）Self-Probing**，式 (2)：

$$(P_t,Q_t)\sim\pi_\theta(\cdot\mid o_t) \tag{2}$$

$P_t=\{p_t^1,\dots,p_t^n\}$ 是 **confirmed patterns（已确信的模式）**，要求写成**通用规则**（因果、前提、约束）而非眼前现象；$Q_t=\{q_t^1,\dots,q_t^n\}$ 是 **open questions（开放疑问）**，其答案会**改变下一步动作选择**（提示词见附录 D）。**这是全文支点**：目标内容第一次由**策略自身的世界模型状态**决定，而非由数据里那条 transition 决定。

**（第二步）Transition Retrieval**，式 (3)：

$$\mathcal{R}_t=\bigcup_{r\in P_t\cup Q_t}\mathrm{Retrieve}(r,\mathcal{T};K) \tag{3}$$

对每个命题 $r$ 独立检索 $K$ 条再取并集，检索器用 **MMR**（Carbonell & Goldstein, 1998）。附录 E 超参：先按相似度取召回池 $R=15$，再贪心选 $K=3$，打分为

$$\mathrm{score}(c)=\lambda\langle q,c\rangle-(1-\lambda)\max_{s\in S}\langle c,s\rangle,\qquad \lambda=0.5$$

$\langle\cdot,\cdot\rangle$ 是归一化嵌入的内积（FAISS inner product），$q$ 为命题嵌入，$S$ 为已选集合，第二项惩罚与已选记录重复的候选；嵌入模型 Qwen3-Embedding-0.6B。检索时**排除同 episode 且步数重叠的迁移**，但**不跨命题去重**。**为什么要检索**：当前状态能试的动作有限（$\mathcal{I}_t$ 不够），答案可能藏在别的轨迹的同一机制里。

**（第三步）Dynamics Synthesis**，式 (4)：

$$y_t=f_{\text{syn}}(o_t,P_t,Q_t,\mathcal{I}_t,\mathcal{R}_t) \tag{4}$$

$f_{\text{syn}}$ 是**外部 instruction-following 模型**（第 5.1 节指定 Qwen3-30B-A3B-Instruct-2507），输入当前上下文、信念与疑问、即时迁移、检索证据。指令要求它 ① 纠正**无证据支持**的 confirmed pattern；② 在证据允许时回答 open question；③ 总结与任务目标和动作选择最相关的动力学。产物 $y_t$ 是需写成知识陈述、不引用证据记录、不含动作建议的自然语言段落。**为何用外部模型**：策略本身可能就错，外部模型扮演"有证据的教师"（代价见第七节）。

**（第四步）Agent-authored 世界模型目标**，式 (5)：

$$\mathcal{L}_{A}(\theta)=-\sum_{(o_t,y_t)\in\mathcal{D}_A}\log p_\theta(y_t\mid o_t) \tag{5}$$

$\mathcal{D}_A=\{(o_t,y_t)\}$ 是合成数据集。**它在做什么**：把式 (1)"条件于动作、预测下一观测"的监督，换成"只给定上下文、生成决策导向的动力学描述"；更新后的参数用于初始化后续 IL 与 RL。注意式子不对称：式 (1) 条件是 $(o_t,a_t)$，式 (5) 只有 $o_t$——AAWM 学的不是"这个动作会怎样"，而是"在这个局面下我该理解什么"。

### 4.3 训练流程按步骤

1. **选轨迹**：从 AgentTraj-L 挑出能在 GiGPO 开源评测代码里**从头执行到尾并成功**的完整轨迹，得 $\mathcal{D}$（ALFWorld 1,655 条 / 19,039 步；WebShop 1,027 条 / 4,971 步）。
2. **建迁移池 $\mathcal{T}$**：每个动作步 = 日志动作 + 3 个额外动作（ALFWorld 从可行动作集去掉日志动作与 `help` 后均匀采样；WebShop 用策略温度 0.8 采 2 个 + 可行集均匀采 1 个），动作规范化去重。池规模：ALFWorld 76,156 / WebShop 19,884 条。
3. **构造目标**：IWM 用 $\bar o_{t+1}$（ALFWorld 用原始观测文本，WebShop 用离线摘要）；AAWM 走式 (2)→(3)→(4) 得 $y_t$。同一尺度下**两者共享 $\mathcal{D}$ 与 $\mathcal{T}$**。
4. **世界模型 SFT**：全参数微调，lr $1\times10^{-5}$，有效 batch 16 / 8，**两个 epoch**。
5. **轻量 imitation learning**：$\mathcal{D}$ 中 32 条轨迹（ALFWorld 429 步 / WebShop 158 步），**一个 epoch**，监督生成 `<action>...</action>`。
6. **GRPO**：prompt 2048 / 4096、response 512、每 episode 最多 50 / 15 步；奖励成功 10、失败 0、非法动作 $-0.1$；actor lr $1\times10^{-6}$；group size 8、每轮 16 组 = 128 环境；KL 系数 0.01。三个种子报 mean±std。

### 4.4 相较基线的改动点具体在哪一处

**不是换模型、不是改 RL、不是加模块，而是只改 SFT 阶段的监督文本 $y_t$。**

| | IWM（基线） | AAWM |
|---|---|---|
| 目标 | $\bar o_{t+1}=g(o_{t+1})$，由环境返回内容决定 | $y_t=f_{\text{syn}}(o_t,P_t,Q_t,\mathcal{I}_t,\mathcal{R}_t)$，由策略信念/疑问决定 |
| 输入条件 | $(o_t,a_t)$ | $o_t$（加检索证据） |
| 信息来源 | 当前这一条 transition | 当前 transition + **跨轨迹检索**的 $\mathcal{R}_t$ |
| 数据、迁移池、IL、GRPO 预算 | — | **与 IWM 完全一致** |

第 5.1 节强调两者在每个尺度下共享 $\mathcal{D}$ 与 $\mathcal{T}$，并在 WebShop 上共用同一构造模型，目的就是把差别锁死在 **target content** 上。

## 五、实验怎么做的

**环境与指标。** ALFWorld（文本具身六类家务，报每类与 All 成功率）；WebShop（HTML 购物，报 Succ 与 Score）；协议沿用 GiGPO 开源仓库（Feng et al., 2026）。另有 AgentGym 四环境：TextCraft、SciWorld、WebShop、ALFWorld。

**基线与模型。** ① 提示式：GPT-4o、Gemini-2.5-Pro、Qwen2.5（1.5B/7B）、ReAct、Reflexion；② 训练式：**Base**（无世界模型初始化）、**IWM**；③ 多环境实验另加 **IWM-SUMMARY**（用与 AAWM 相同的 Qwen3-30B-A3B 先摘要再预测），控制构造模型能力。策略为 Qwen2.5-1.5B/7B-Instruct；检索嵌入 Qwen3-Embedding-0.6B；构造模型 Qwen3-30B-A3B-Instruct-2507（**仅用于构造目标**）。

**主要结果（表 1 "IL + RL initialized by World Modeling" 区块）。**

- 1.5B：ALFWorld All — Base **74.0**±4.6 → IWM **76.8**±2.8 → AAWM **83.1**±1.7；WebShop Score 71.1 → 77.0 → **81.5**，Succ 59.4 → 67.2 → **73.4**。
- 7B：ALFWorld All 84.6±1.3 → 86.7±2.0 → **90.1**±2.0；WebShop Score 80.8 → 82.2 → **84.2**，Succ 70.6 → 73.2 → **76.6**。
- 提示式对照：GPT-4o 48.0 / 23.7（ALFWorld All / WebShop Succ）；Gemini-2.5-Pro 60.3 / 35.9。
- **难度分层**：1.5B 相对 IWM：Cool **+11.0**、Pick2 **+10.6**、Look **+7.5**；7B：Look **+9.0**、Cool **+7.4**、Pick2 **+6.0**；IWM 仅在 1.5B 的 Pick 与 7B 的 Clean 上略高。

**消融（表 2，1.5B，去掉 GRPO，所有行共用同一 imitation learning 阶段）。** ALFWorld All / WebShop Succ：Base 12.0±2.6 / 4.2±1.2 → AAWM **24.7**±2.6 / **10.7**±1.2 → 去 Transition Retrieval 19.3±2.4 / 7.5±1.6 → 再去 Self-Probing 18.0±1.6 / 7.0±3.1。作者解读：没有检索只见即时迁移，不足以解答开放疑问；没有 Self-Probing 则没有"目标该覆盖什么"的信号。

**更多环境（表 3，AgentGym 四环境，1.5B，纯监督训练）。** 训练集混合 TextCraft/SciWorld/WebShop/ALFWorld（4 个环境共 8,844 条轨迹；评测 200/200/200/100 任务）。加权 All：Qwen2.5 提示式 13.0；Base 61.1±1.8；IWM 59.6±2.4（SciWorld 79.8，但 WebShop 掉到 83.0、ALFWorld 掉到 30.2）；IWM-SUMMARY 61.4±2.0（ALFWorld 暴跌到 20.0）；**AAWM 69.3±1.3，是唯一在四个环境都超过 Base 的世界模型初始化**，最大单项增益 SciWorld **+21.0**。

**分析（第 6 节）。**

- **训练动力学**：AAWM 收敛更快且**保持最高熵**（图 3/4，EMA 0.95），说明不是坍缩而是更有效探索；回答更长——WebShop **167.6** vs IWM **121.4** tokens，ALFWorld **98.7** vs **57.8**。
- **对齐性（表 4，LLM-as-judge）**：决策有用性 $\bar d_1$ = 0.96 / 1.00；纠正错误信念 $\bar d_2^P$ 仅 **0.19 / 0.25**；解答开放疑问 $\bar d_2^Q$ = 0.96 / 0.99（ALFWorld / WebShop）。
- **命题数量（表 5）**：$n=1$ → 10.7 / 7.8；$n=3$ → 24.7 / 10.7；$n=5$ → 26.0 / 11.7，故取 $n=3$。
- **无 IL 的纯 RL（表 6，1.5B）**：Base 71.6±2.8 / 58.6±3.8；IWM 崩到 **13.2**±3.2 / 54.2±2.7；AAWM **70.6**±2.7 / **63.8**±2.7。

> 注意：提取文本的表格单元格按绘制顺序竖排；我已按列标题逐列还原并与正文数字互校（74.0→76.8→83.1、Cool +11.0 等均吻合），**对应关系可靠**。

## 六、优点与贡献

1. **新问题**：把世界模型训练的目标内容明确成 *target selection*，给出"环境驱动 vs 决策驱动"判据；可迁移到"奖励设计写什么""critic 学什么"。
2. **新方法**：用"自陈信念与疑问 → 跨轨迹检索证据 → 外部模型合成目标"三步，把 target 从"复述下一观测"变成"回答策略此刻的疑问"；不改架构与 RL，极易插入现有 SFT→RL 管线。
3. **对照设计严谨**：同一份 $\mathcal{D}$、$\mathcal{T}$、同一构造模型与 IL/GRPO 预算，把收益逼到 target content 上；多环境实验另加 IWM-SUMMARY 控制构造模型能力。
4. **新分析**：除成功率外报 entropy、response length、LLM-as-judge 与无 IL 的纯 RL 对照，说明更快收敛不是坍缩而是探索。
5. **负结果有信息量**：IWM 无 IL 时崩到 13.2%，AAWM 保持 70.6%——**下一观测预测作为初始化目标本身可能是有害的热身**。

## 七、局限与可疑之处

**作者自承的局限。** Self-Probing **只在数据构造时跑一次**，目标只是"那一刻"的信念快照（作者提出可每轮重建形成 curriculum，留作未来工作，代价是构造算力）；实验**只覆盖纯文本环境**；伦理上承认合成目标可能反映**不完整或有偏**的迁移证据，使智能体学到错误动力学。

**我认为可疑或没讲清楚的地方。**

1. **构造模型的能力混淆（最大疑点）。** $y_t$ 由 **Qwen3-30B-A3B** 合成，策略只有 1.5B/7B。WebShop 与 AgentGym 都用 IWM-SUMMARY 控制过构造模型能力，但**主结果表 1 的 ALFWorld 没有这一行**（该环境的 IWM 目标是原始观测文本，不需构造模型）。于是 ALFWorld 上无法排除更简单的解释：**这是把 30B 的环境知识蒸馏进了小策略**，"决策导向"只是伴随属性；表 3 缓解但未消除该疑问。
2. **消融差距偏小且只有 1.5B。** 24.7 → 19.3 → 18.0 的 std 分别 2.6 / 2.4 / 1.6，**"去检索"与"去 probing"间 1.3 点落在噪声量级内**，却支撑"两个组件都必要"；且**无 7B 消融**。
3. **裁判 = 教师，自偏好风险。** 表 4 的 judge 与生成 $y_t$ 的模型**同名同版本**（Qwen3-30B-A3B-Instruct-2507），$\bar d_2^Q$ 近乎满分可能部分是"自己评自己"；而 $\bar d_2^P$ 只有 0.19/0.25，说明**"纠正错误信念"这条机制绝大多数时候没触发**。
4. **收益随模型变大而缩小、成功率与奖励分不同向。** 1.5B 上 AAWM 相对 IWM 的 gain 是 ALFWorld +6.3 / WebShop Succ +6.2；**7B 上两者都只有 +3.4**，Abstract 取的是小模型最好情形。此外表 2 只报 Succ，而表 1 的 IL-only 区块中 AAWM 的 Score（25.7）其实**低于 Base（26.6）**，只有 Succ 更高——论文未讨论这种不一致。
5. **只用成功轨迹，检索池与训练池同源。** $\mathcal{D}$ 只保留成功轨迹，失败模式没进 target；检索虽排除同 episode 重叠步，但**证据仍来自将被 imitation learning 使用的同一批轨迹**，存在泄漏可能，论文无留出集检索对照。
6. **排版层面。** 附录 D 提示词先写 "each with 2 to 3 statements"，随后又要求 "output exactly {max_beliefs} ..."；式 (3) 正文用 $\mathcal{R}_t$、图 2 用 $R_t$——记号不一致，逐字引用请对照 PDF。**除此之外未发现异常内容。**

## 八、与同批论文的关系

本批 10 篇：1008_SCA、15751_HGPO、18_PA-MoE、22669_GraphGPO、35074_KPO、1406_CITA、14071_AgentBrew、1423_CIPO、1432_PhGPO、13085_AAWM。

**先声明**：本次只允许读本文件的 txt，**其余 9 篇我一个字都没读**，跨篇判断一律标为待核实。

- **共享问题设定（承继/共鸣，待核实）**：本篇主张"只靠下一观测或环境恰好暴露的内容做监督太粗糙"。若 15751_HGPO、22669_GraphGPO、1432_PhGPO、35074_KPO、1423_CIPO 中若干篇属"把粗粒度结果信号细化"的路线，则二者**同一诊断、不同处方**（改 RL 信号用法 vs 改 SFT 监督内容），**互补而非竞争**。
- **共享技术底座（本篇内可证实）**：AAWM 的 GRPO **直接沿用 GiGPO 开源实现**（Feng et al., 2026）；**若本批有研究 group-based / hierarchical RL 的论文，二者在"同一套 group-relative RL 基础设施 + 改监督信号"上高度可比**，是天然的对比接口（GiGPO 与 "HGPO"/"GraphGPO" 字面接近，**相似性待核实**）。此外同批工作若也用 ALFWorld/WebShop/AgentGym，**可把 $y_t$ 构造流程当作"换 SFT 目标"的插件直接接上**。
- **总体判断**：AAWM **方法上相对独立**（改数据构造而非优化算法/架构），但其判据与前几篇"结果级监督太粗"的关切**思路相通**；对比时可用一条轴排开：**信号发生在哪一层**（SFT 目标内容 / RL 信用分配 / 模型结构）。

## 九、如果我要接着做，可以做什么

**方向 1：补"构造模型能力"对照，把增益拆成"知识蒸馏"与"决策导向"。**
- **改什么**：主实验 ALFWorld 上加 **IWM-SUMMARY** 行（用与 AAWM 相同的 Qwen3-30B-A3B 把 $o_{t+1}$ 摘要成"动力学风格"段落后再让策略预测），再加 **AAWM-SHUFFLE** 行（把 $y_t$ 换成无关轨迹的合成目标，保留"30B 写、决策风格"属性）。
- **为什么有信息**：若三者接近，收益来自"30B 写的文本"而非"决策导向"；只有 AAWM 明显高而 SHUFFLE 明显低，才是对本文主张的干净支持。
- **第一步**：固定 Qwen2.5-1.5B 与全部数据/预算（照 Appendix A.1），只换 target 跑表 1 的 IL+RL 区块，报 All 与 Cool/Look/Pick2。

**方向 2：把 Self-Probing 做成随训练演化的课程，并度量"目标漂移"。**
- **改什么**：每轮 GRPO 后用**当前策略**重新 Self-Probing，并把本轮**失败轨迹的上下文**作为 probing 输入重建 $y_t$；记录各轮 $y_t$ 的嵌入漂移量与新增命题类型。
- **为什么可能有效**：现有监督是"一次性快照"，而策略的疑问会随能力变化（先问前提条件、后问对象持久性），让目标跟着走等价于自动课程学习。
- **第一步**：ALFWorld 1.5B 上比较"1 次构造 / 每轮重建 / 每两轮重建"，**控制总构造 token 与训练步数一致**，报 success、entropy 与漂移量。

**方向 3：给合成目标加可执行自检，压掉幻觉并量化比例。**
- **改什么**：把 $y_t$ 拆成句子级断言，半自动翻译成环境可执行的动作对（如断言"用 sinkbasin 能洗掉手上的土豆"→ 执行 `clean potato 1 with sinkbasin 1`），逐条验证，**只保留被验证的断言**用于 SFT，并报告 hallucination rate。
- **为什么可能有效**：$\bar d_2^P$ 仅 0.19/0.25，说明大量错误信念未被纠正、可能被原样写进 $y_t$；环境是免费的 oracle，比 LLM judge 可靠。
- **第一步**：从 ALFWorld 与 WebShop 各抽 200 个 $(o_t,q,y_t)$，测"可执行比例"与"与断言一致的比例"；再用过滤后的 target 重跑表 2 设置。

## 十、自测问题

1. **（方法内部）** 式 (1) 条件是 $(o_t,a_t)$，式 (5) 只有 $o_t$。丢掉动作条件 $a_t$ 在语义上意味着什么？为什么这与"决策导向"的主张一致？
2. **（方法内部）** MMR 打分为 $\lambda\langle q,c\rangle-(1-\lambda)\max_{s\in S}\langle c,s\rangle$。若把 $\lambda$ 改成 1.0 会怎样？作者为什么让召回池 $R=15$ 大于最终 $K=3$？
3. **（方法内部）** Dynamics Synthesis 指令里有 "DO NOT include action suggestions or plans"。若删掉这句，训练目标会变成什么？作者为什么要禁止？
4. **（实验设计）** 表 2 中"去 Transition Retrieval"只比完整版低 5.4 点，"去 Self-Probing"再低 1.3 点。请指出该消融在**显著性与规模覆盖**上的两个问题，并设计一个能分离两个组件贡献的实验。
5. **（实验设计）** 论文用 Qwen3-30B-A3B-Instruct-2507 同时做目标构造与 LLM-as-judge 裁判，这为何可能让表 4 指标偏高？给出一个不依赖该模型的替代验证方案。
