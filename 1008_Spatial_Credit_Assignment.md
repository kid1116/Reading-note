# 1008 Spatial Credit Assignment（SCA）精读笔记

> 原文标题：**Spatial Credit Assignment: Exploiting Rollout Geometry for GUI Agent Reinforcement Learning**
> 依据文件：`_tools/txt/1008_Spatial_Credit_Assignment.txt`（3691 行，含正文、参考文献、附录 A.1–A.14 与 NeurIPS Checklist）。

本文指出 GRPO、RLOO、DAPO 等组策略优化方法的基线只由标量奖励构成，丢弃了采样动作的几何位置；在 GUI grounding 这类度量动作空间里，这造成"空间信用坍缩"——近失误与远失误同分，全失误组梯度归零。作者提出 SCA：用留一法方向中位斜率回归拟合局部奖励趋势，按相关系数软门控与原始基线融合；全失误组另加邻近辅助项。五个基准上全面超过 GUI-R1，单步开销仅 +2.5%。

---

## 一、这篇论文的一句话贡献

**作者挑明"组基线只能依赖奖励"这一隐含假设（Remark 1），论证它是空间信用坍缩的根因，并给出免 critic、动作条件化的代理优势估计器 SCA：只替换组基线，不动目标函数与裁剪规则，即在五个基准上稳定超过 GUI-R1。**

---

## 二、读之前需要的前置知识

**1. 策略梯度中的基线。** REINFORCE 方差大，减去不依赖动作 $a$ 的 $b(x)$ 后方差下降而期望不变。**无偏性完全依赖 $b$ 不依赖 $a$。** Eq. (3) 正利用这点：基线一旦变成 $\hat R_{-i}(a_i)$，无偏性即破坏，SCA 遂为"故意有偏"的代理估计器。

**2. 组策略优化。** GRPO 用组均值 $\bar r$，RLOO 用留一均值 $\bar r_{-i}=\frac{1}{N-1}\sum_{j\ne i}r_j$，REINFORCE++ 用上下文价值 $\hat V(x)$，DAPO 保留 GRPO 基线而改裁剪与采样。Eq. (1) 把它们统一，**SCA 就是往这个式子的 $b_i$ 里插东西，故可即插即用。**

**3. 度量动作空间与稀疏命中奖励。** GUI agent 动作是坐标 $a\in\mathcal{A}\subseteq\mathbb{R}^2$，奖励 $R_{\text{bin}}(a;x)=\mathbb{1}[a\in T(x)]$。因 all-miss 组所有 $r_i=0$，任何只依赖 $\{r_j\}$ 的基线都给出 $\hat A_i\equiv0$，整组梯度为零——哪怕这 5 个点明显偏向目标右侧。这就是 spatial credit collapse。

**4. 稳健回归与击穿点。** OLS 用一个离群点即可拉偏斜率；Theil–Sen（中位斜率）取所有点对斜率的中位数，$N-1=4$ 点击穿点约 29%。**$N=5$ 时二维仿射需 3 参数只剩 4 点，故降到一维投影 + 中位斜率，这是 SCA 最核心的工程决策**；$c_{MS}$ 即稳健化代价常数。

**5. 残差平方和 RSS。** $\text{RSS}(f;\mathcal{A})=\sum_i(r_i-f(a_i))^2$ 衡量预测器对组内奖励的拟合误差。因 $B_i$ 无闭式界，作者不证策略梯度方差界而证 RSS 下降：拟合误差小意味着代理基线更贴近真实局部奖励地形。

---

## 三、研究背景与要解决的问题

**前人怎么做。** 论文分三条线：

- **组策略优化线**：GRPO、RLOO、REINFORCE++、DAPO。**Remark 1** 指出该家族每个成员的 $b_i$ 都只依赖 $\{r_j\}$ 和（REINFORCE++ 的）$x$，**没有一个依赖 $\{a_j\}$**。
- **连续控制的动作相关基线线**：Q-PROP、Stein identity 控制变量、factorized baselines，以及 Tucker et al. 关于"动作相关基线常是海市蜃楼"的负面结果。这条线已做动作条件化，但**普遍依赖学出来的 critic 或解析控制变量假设**。
- **奖励塑形线**：GUI-G²、SE-GUI、RSGround-R1 用距离或几何感知奖励缓解稀疏反馈，有效但**作用在监督层**：塑形奖励算完后，组方法依然从奖励值而非 rollout 联合几何构造基线。

**具体局限（两种失效模式）**，作者称其为任何动作无关 $b$ 都继承的结构性后果：**(1) mixed-hit 组里所有 miss 拿到相同优势**，因 $r_i\in\{0,1\}$ 单独就决定了 $\hat A_i$；**(2) all-miss 组里 $\hat A_i\equiv0$**，即使 rollout 排布已明确指示改进方向。故"要消除空间信用坍缩，基线必须依赖 $\{a_j\}$"。

---

## 四、方法详解（核心）

### 4.1 整体框架与数据流

SCA 不改策略目标、裁剪与采样器，**只改"构造基线 $b_i$ 所用的信息"**。流程见 Algorithm 1：给定组 $G$（$N=5$）与目标 $T(x)$，先算 regime 指示器，再分三条支路：

- `ALLHIT`（$\min_j r_j>1-\tau_{\text{hit}}$）：$\hat A_i\leftarrow0$，让位族基线；
- `ALLMISS`（$\max_j r_j<\tau_{\text{hit}}=0.05$）：走 proximity 辅助支路（Eq. 9）；
- `MIXED`：走方向中位斜率 LOO 集成 + 相关门控（Eqs. 5–8），核心创新。

**`MIXED` 在优势层（advantage layer），`ALLMISS` 在监督层（supervision layer）**——此分层用以说明 SCA 与奖励塑形正交可叠加。

### 4.2 统一形式与问题形式化

统一组基线（Eq. 1）：

$$
\hat A_i=\frac{r_i-b_i(G;x)}{\sigma(r)+\epsilon},\qquad G=\{(a_j,r_j)\}_{j=1}^{N} \tag{1}
$$

$r_i$ 为第 $i$ 条 rollout 奖励，$b_i$ 为基线，$\sigma(r)$ 为组内奖励标准差，$\epsilon$ 防除零。**作用：把整个家族收进同一模板，使"基线能否看到动作"成为唯一变量。** SCA 把 $b_i$ 换成：

$$
\tilde A_i^{\text{SCA}}=r_i-\hat R_{-i}(a_i) \tag{2}
$$

$\hat R_{-i}(a_i)$ 用 $\{(a_j,r_j)\}_{j\ne i}$ 拟合、在留出的 $a_i$ 上求值。因它依赖 $a_i$，无偏性被破坏，产生偏差（Eq. 3）：

$$
B_i=\mathbb{E}_{a_i\sim\pi_\theta(\cdot|x)}\left[\nabla_\theta\log\pi_\theta(a_i\mid x)\cdot\hat R_{-i}(a_i)\right]\neq 0 \tag{3}
$$

**为何这样设计**：SCA 是**有偏代理而非无偏控制变量**，用更低的逐 rollout 拟合误差换偏差。$B_i$ 无闭式界，偏差控制交给 §4.4 的门控 $w$：相关弱时 $w\to0$，退化为原族方法。

### 4.3 局部奖励趋势与有限样本分析

**Assumption 1（局部线性平滑）**：存在 $w^\star\in\mathbb{R}^2$、$b^\star\in\mathbb{R}$ 使 $r_i=w^{\star\top}a_i+b^\star+\xi_i$；$\xi_i$ 为局部仿射残差（含真实奖励偏离线性部分与采样随机性），条件零均值、参数 $\sigma$ 次高斯。作者点明：**奖励无关基线家族对应 $w^\star=0$，这在 $T(x)$ 边界附近几乎从不成立**——这等于说"奖励无关基线"隐含"梯度恒为零"。

**Theorem 1（有限样本残差界，Eq. 4）。** 固定 $\mathcal{A}=(a_1,\dots,a_N)$，对单位方向 $e_d$ 定义杠杆臂与最小投影间隔：

$$
L_d(\mathcal{A}):=\max_i\left|\langle e_d,a_i\rangle-\operatorname{med}_j\langle e_d,a_j\rangle\right|,\qquad
\Delta_d(\mathcal{A}):=\min_{j\neq k}\left|\langle e_d,a_j-a_k\rangle\right|
$$

设 $\Delta_d(\mathcal{A})>0$，$\hat R^{\text{ens}}_{-i}$ 为方向权重 $\alpha_d\propto\hat\rho_d^2$ 的集成中位斜率 LOO 预测器，$\hat\rho^2_{\text{ens}}:=\sum_d\alpha_d\hat\rho_d^2\in[0,1]$。则 $N\ge4$ 时：

$$
\mathbb{E}_\xi\!\left[\text{RSS}\!\left(\hat R^{\text{ens}}_{-i};\mathcal{A}\right)\Big|\mathcal{A}\right]
\;\lesssim\;
\left(1-\hat\rho^2_{\text{ens}}\right)\text{RSS}\!\left(\bar r_{-i};\mathcal{A}\right)
\;+\;
c_{MS}\,\frac{\sigma^2}{N}\max_d\left(\frac{L_d(\mathcal{A})}{\Delta_d(\mathcal{A})}\right)^{2} \tag{4}
$$

- 左边是集成预测器的**样本内**期望 RSS（非策略梯度方差）；
- 第一项：投影与奖励相关时拟合误差按 $1-\rho^2$ 下降；
- 第二项：**中位斜率替代 OLS 的稳健性代价**，两项均无量纲；只要投影后位置不几乎重合就小；
- $\lesssim$：允许一个未确定低阶项（来自 $\alpha_d$ 与逐方向预测器的相互依赖）。

**证明思路（A.2）**：Step 1 给单方向界（斜率方差乘 $L_d^2$）；Step 2 用 Jensen 把凸权重挪进残差平方内，代回 Eq. (7)。

**经验验证（$N=5$，前 1000 组）**：加性项相对 $\text{RSS}(\bar r_{-i})$ 中位数 **0.08**、95 分位 **0.21**；$(1-\hat\rho^2_{\text{ens}})$ 在 $\hat\rho_{\text{ens}}>0.3$ 的组上中位数 **0.42**，故**残差缩减约为加性代价的 5 倍**；$\lesssim$ 松弛项实测为右侧高估残差中位数 **1.07×**（95 分位 1.18×，约 10%）。该界**不管策略梯度方差**。

### 4.4 方向中位斜率集成与相关门控

$N=5$ 时二维仿射需 3 参数而留一后仅剩 4 点，病态；**故降到一维**：在 $e_d\in\{0^\circ,45^\circ,90^\circ,135^\circ\}$（$D=4$）上分别做 Theil–Sen 式中位斜率回归（Eq. 5）：

$$
\hat w_d=\operatorname{median}\left\{\frac{r_j-r_k}{\langle e_d,\,a_j-a_k\rangle}\;:\;j,k\neq i,\;j<k\right\},\qquad
\hat b_d=\operatorname{median}\left\{r_j-\hat w_d\langle e_d,a_j\rangle\right\} \tag{5}
$$

$\hat w_d$ 是沿方向 $d$ 的中位两两斜率（比 OLS 稳健），$\hat b_d$ 为残差中位数，$j,k\neq i$ 即留一。一维拟合仅 2 参数，$N-1=4$ 时良定，击穿点约 **29%**。各方向信号量取决于局部梯度朝向，故按平方相关加权集成：

$$
\hat R_{-i}(a_i)=\sum_{d=1}^{D}\alpha_d\left(\hat w_d\langle e_d,a_i\rangle+\hat b_d\right),\qquad
\alpha_d=\frac{\hat\rho_d^{2}}{\sum_{d'}\hat\rho_{d'}^{2}} \tag{6}
$$

$$
\hat\rho^2_{\text{ens}}=\sum_{d=1}^{D}\alpha_d\hat\rho_d^{2}\in[0,1] \tag{7}
$$

$\hat\rho_d$ 是投影位置与奖励在 $N-1$ 个 LOO 点上的 Pearson 相关系数；$\alpha_d$ 是归一化凸权重；$\hat\rho^2_{\text{ens}}$ 写成凸组合以**与 Theorem 1 对齐**。记 $\hat\rho_{\text{ens}}:=\sqrt{\hat\rho^2_{\text{ens}}}$。

**门控用软混合而非硬阈值**（Eq. 8）：

$$
\hat R^{\text{gated}}_{-i}(a_i)=w\cdot\hat R_{-i}(a_i)+(1-w)\cdot b_{\text{family},i}(G;x),\qquad
w=\operatorname{clamp}\!\left(\frac{\hat\rho_{\text{ens}}-\rho_{\text{lo}}}{\rho_{\text{hi}}-\rho_{\text{lo}}},\,0,\,1\right) \tag{8}
$$

$b_{\text{family},i}$ 为族基线（GRPO $\bar r$，RLOO $\bar r_{-i}$，REINFORCE++ $\hat V(x)$）；全文固定 $[\rho_{\text{lo}},\rho_{\text{hi}}]=[0.3,0.7]$。**理由是硬阈值会让组在边界附近跳变。** 混合命中组最终优势：

$$
\hat A_i^{\text{SCA-mix}}=\frac{r_i-\hat R^{\text{gated}}_{-i}(a_i)}{\sigma(r)+\epsilon}
$$

当 $\hat\rho_{\text{ens}}\to0$ 时 $w\to0$，SCA **精确退化为原族方法**，$B_i$ 随之消失。

### 4.5 全失误组与门控邻近辅助

all-miss 组里 Eq. (5) 的预测器退化为常数，$\hat A_i^{\text{SCA-mix}}\equiv0$；但 rollout 到 $T(x)$ 的距离仍有差异，故编码为辅助优势（Eq. 9）：

$$
\text{prox}_i=\exp\!\left(-\frac{d\!\left(a_i,\,c(T(x))\right)}{k_{\text{prox}}\cdot\operatorname{diam}(T(x))}\right),\qquad
\hat A_i^{\text{SCA-miss}}=\alpha_{\text{eff}}\cdot z(\text{prox})_i \tag{9}
$$

$$
k_{\text{prox}}=k_{\text{base}}+\frac{k_{\text{scale}}}{\operatorname{diam}(T(x))},\qquad
\alpha_{\text{eff}}=\alpha_{\text{max}}\cdot\operatorname{clamp}\!\left(\frac{\sigma(\text{prox})}{\sigma_{\text{ref}}},0,1\right)
$$

$c(T(x))$、$\operatorname{diam}(T(x))$ 是目标质心与对角线长；$z(\cdot)$ 为组内 z 标准化；$k_{\text{prox}}$ 是**尺度自适应带宽**；$\alpha_{\text{eff}}$ 是**判别性扩散门控**，近邻分数无区分度时关掉整条支路。全失误判定用 $\max_j r_j<0.05$ 而非精确等于 0，以在连续塑形奖励下生效（S6：精确相等会静默关掉 GUI-G² 场景下的该支路）。

作者定位诚实：prox 用了**真值 $T(x)$**，注入的是**外部监督**而非组内奖励的重加权，故被当作"与奖励塑形平行的辅助监督"单独归因。它**随策略改进自动退场**：all-miss 组比例从 step 0 的 **54%** 降到 step 348 的 **16%**（A.14）。

### 4.6 训练流程分步

1. **采样**：从 $\pi_{\theta_{\text{old}}}(\cdot|x)$ 采 $N=5$ 条 rollout，解码坐标 $a_j$，按二值命中测试算 $r_j$。
2. **判 regime**：`ALLMISS` if $\max_j r_j<0.05$；`ALLHIT` if $\min_j r_j>1-0.05$；否则 `MIXED`。
3. **all-miss**：按 Eq. (9) 算 $\text{prox}_j$、$\alpha_{\text{eff}}$，$\hat A_i\leftarrow\alpha_{\text{eff}}z(\text{prox})_i$。
4. **all-hit**：$\hat A_i\leftarrow0$。
5. **mixed**：对每个 $i$ 与每个 $d$，在 $\{(\langle e_d,a_j\rangle,r_j)\}_{j\ne i}$ 上拟合 $(\hat w_d,\hat b_d)$，算 $\hat\rho_d$、$\alpha_d$、$\hat R_{-i}(a_i)$、$\hat\rho^2_{\text{ens}}$、$w$，得 $\hat R^{\text{gated}}_{-i}$ 与 $\hat A_i^{\text{SCA-mix}}$。
6. **更新**：把 $\{\hat A_i\}$ 塞进**原封不动的** PPO 式裁剪代理目标反传；优化器、LR 调度、rollout 预算与基线一致（A.4 / Table 5）。

**安全阀 S1–S6**：S1 剔除投影间隔 $<\tau_a=10^{-3}$ 的退化点对（有效点对不足 2 则该方向 $\hat\rho_d=0$）；S2 若 LOO 子集奖励标准差 $<\tau_r=10^{-6}$ 则全部 $\hat\rho_d=0$，$w\to0$ 退回族基线，**自动处理"1 命中 / 4 失误"**；S3 投影或奖励方差过低时 $\hat\rho_d=0$；S4 若 $\sum_d\hat\rho_d^2<\tau_r$ 则 $\hat R_{-i}(a_i):=\bar r_{-i}$；S5 邻近 z 标准化带 $\epsilon_z=10^{-6}$；S6 全失误检测用阈值。$\sigma_{\text{ref}}$ 用**前 500 步流式 P-square 中位数**在线估计。

### 4.7 相较基线的改动点具体在哪

**唯一改动是 Eq. (1) 里的 $b_i$**：mixed 组用 $r_i-\hat R^{\text{gated}}_{-i}(a_i)$，all-miss 组用 $\alpha_{\text{eff}}z(\text{prox})_i$，all-hit 组为 0。目标函数/裁剪、采样器、优化器、LR 调度、rollout 预算与训练数据全部不变（Table 5）。

---

## 五、实验怎么做的

**环境与训练。** 五个基准、三个平台：SCREENSPOT、SCREENSPOT-PRO、GUI-ACT-WEB、OMNIACT、GUI-ODYSSEY，**只用 test split**。主实验 backbone 为 QWEN2.5-VL-3B，缩放用 7B；RFT 用 EasyR1（与基线 GUI-R1 同一框架），跑相同 epoch 数；SFT 基线沿用 LLAMA-FACTORY 一 epoch 协议；推理用同一 zero-shot prompt。所有 RFT 用 $N=5$、标准 PPO 裁剪、相同优化器/学习率/rollout 预算，**唯一差异是 SCA 的两层修改**。超参：$D=4$、$[\rho_{\text{lo}},\rho_{\text{hi}}]=[0.3,0.7]$、$\alpha_{\text{max}}=1.0$、$k_{\text{base}}=0.5$、$k_{\text{scale}}=2.0$、$\tau_{\text{hit}}=0.05$。硬件 8×NVIDIA H800，**348 步**，max_pixels=1,258,291，**每配置单种子**。

**指标与基线。** 指标为 Type（action-type accuracy）、GR（click-grounding accuracy）、SR（step success rate）；SCREENSPOT/-PRO 报 grounding accuracy，GUI-ODYSSEY 以 SR 为 headline。基线含 SeeClick、Os-Atlas-4B/7B、ShowUI-2B、CogAgent-18B、Aria-GUI、UGround-7B、QwenVL2.5-3B/7B*、Claude*、UI-R1-3B、GUI-R1-3B。

**主要结果。** Table 1（SSPRO 六域 Text/Icon，序 Dev/Creative/CAD/Scientific/Office/OS，再 SCREENSPOT Web/Desktop）：SCA-3B = 36.5/5.8、43.1/7.7、28.2/9.4、64.5/19.6、56.9/19.5、31.4/7.9；Web 92.2/75.0、Desktop 95.6/73.8。对照 GUI-R1-3B = 33.8/4.8、40.9/5.6、26.4/7.8、61.8/17.3、53.6/17.0、28.1/5.6；89.6/72.1、93.8/64.8。Table 2：SCA-3B 的 GUIAct Type/GR/SR = 92.71/90.42/82.51，OmniAct-Web 94.57/81.83/81.71，OmniAct-Desktop 95.31/90.52/90.62，GUI-ODYSSEY SR **68.90**；GUI-R1-3B 对应 89.86/87.42/76.31、88.58/75.10/75.08、91.86/78.37/78.31、**64.41**。SCA 在**全部 16 个 SSPRO/SS 子列、9 个 GUIAct/OmniAct 列、以及 GUI-ODYSSEY SR** 上提升；增益 SSPRO/SS **+1.0 到 +9.0**，跨域低层任务 **+2.85 到 +12.31**，GUI-ODYSSEY **+4.49**。1000 次 test-set 重采样的 bootstrap 95% CI：SCA **[67.4, 70.3]**，GUI-R1 **[63.0, 65.8]**（作者注明这**只反映固定策略下的测试集噪声，不是训练种子方差**）。

**两层归因（Table 3）。**

| 方法 | GUI-ODYSSEY SR | Δ | SCREENSPOT-PRO Avg | Δ |
|---|---|---|---|---|
| Baseline | 64.41 | — | 22.9 | — |
| Proximity-only | 67.13 | +2.72 | 23.5 | +0.6 |
| Mix-only | 66.52 | +2.11 | 24.5 | +1.6 |
| **Full SCA** | **68.90** | **+4.49** | **25.0** | **+2.1** |

长程基准上两层**各贡献约一半且近似可加**；纯 grounding 的 SSPRO 上 Mix-only 已拿到 Full SCA 增益的 **76%**。

**组件消融（Table 4，列序 SS-Avg/SSPro/GUIAct-GR/GUIOdy/OmniD-GR/OmniW-SR）。** Full SCA = 83.7/25.0/90.4/68.9/90.5/81.7，**每列最优**；w/o adaptive α = 83.0/24.4/90.0/64.5/89.6/79.6；w/o soft ρ-blend = 82.1/23.7/89.3/65.7/88.4/77.0；w/o proximity = 82.5/24.5/89.6/66.5/88.9/78.3；w/o ensemble LOO = 82.2/23.5/89.4/**62.8**/89.1/78.5；w/o LOO trigger = 82.6/24.1/89.7/64.0/89.2/79.7。**去掉 ensemble LOO 或 LOO trigger 在 GUI-ODYSSEY 上会掉到 GUI-R1 以下**，说明长程误差累积时趋势支路必需。Table 12 逐组件差值：Ensemble LOO 在 GUIOdy 上 **+6.1**、LOO trigger +4.9、Adaptive α +4.4、Soft ρ-blend +3.2、Proximity +2.4。

**可组合性（Table 6，2×2 因子）。** SSPRO：GUI-R1+Binary 22.9，SCA+Binary 25.0，GUI-R1+GUI-G² 27.6（+4.7），SCA+GUI-G² **31.8（+8.9）**；纯加性预期 +6.8，**实际超出 +2.1，即正交互**。OMNIACT-DESKTOP GR：78.4 / 90.5 / 81.1 / 93.8（+15.4），加性预期 +14.8，交互 +0.6。机制侧：GUI-G² 把 mixed-hit 组 $\hat\rho^2_{\text{ens}}$ 中位数从 **0.18** 提到 **0.41**，median $w$ 从 **0.0** 提到 **0.28**。

**即插即用与缩放。** RLOO drop-in（Table 7，令 $b_{\text{family},i}=\bar r_{-i}$）：80.3/22.5/78.0/63.8 → 82.0/24.4/89.4/67.6，Δ **+1.7/+1.9/+11.4/+3.8**（GRPO 口径 +2.8/+2.1/+12.1/+4.49）。7B（Table 10）：SCA-7B = 86.4/29.8/84.62/84.13/92.84/72.45，**六个汇总指标全部超过 GUI-R1-7B**（82.6/25.4/80.31/77.35/83.33/65.83）；3B→7B 增 +2.7、+4.8、+3.55。敏感性（Table 8）：门控带与 $D\in\{2,4,6,8\}$ 所有配置与默认差值 **≤1.1（SS-Avg）/≤1.9（GUI-ODYSSEY）**。开销（Table 9）：median-slope LOO **117 ms（2.4%）**、proximity **6 ms（0.1%）**；总步 4822 ms vs. vanilla GUI-R1 4702 ms，即 **+2.5%**（100 步均值，std 0.3%）。

**训练侧诊断（A.14）。** EGR（产生非零非退化优势的组占比）SCA 从约 **95%** 衰减到约 **30%**，GUI-R1 从约 **50%** 到约 **22%**；step 0–200 的高学习信号阶段 SCA 保持 **2–2.5×** EGR 优势。Figure 4：SCA 用约 **60%** 步数达到 GUI-R1 最终奖励水平。Figure 5：all-miss **54%→16%**，all-hit **1%→65%**，作者称这是门控自然产生、**非手工调度**。

**一次自我否证（A.12）**：作者原假设 SSPRO 目标更细长导致收益波动，实测长宽比 AR$=\max(w,h)/\min(w,h)$：SCREENSPOT（$N=1272$）中位 **1.70**、均值 3.47；SSPRO（$N=1581$）中位 **1.71**、均值 3.92——**假设被自己的数据证伪**。

---

## 六、优点与贡献

1. **【新问题】** 命名并形式化 spatial credit collapse，并与"奖励稀疏"区分：根因是**组策略优化内部缺少动作条件化信用分配规则**，Remark 1 论证这是整个家族的结构性后果。属问题定义级贡献。
2. **【新方法】** 把基线由 $b(\{r_j\})$ 推广为 $b(\{(a_j,r_j)\})$。最值得学的是**工程与理论咬合**：$N=5$ 逼出"一维投影 + 中位斜率 + 按 $\hat\rho_d^2$ 集成"，超参都能在 Theorem 1 的两项中找到解释。
3. **【新分析】** Theorem 1 把"方差缩减项"与"稳健化代价"分开并**用实测量化**，且**主动声明它不是策略梯度方差界**。
4. **【可迁移性】** 只动 $b_i$，目标/裁剪/采样器全不变，RLOO drop-in 也正向，开销仅 2.5%。低侵入 + 低开销是能否被采用的决定性因素。
5. **【诚实归因】** 提前堵住"增益全靠用了真值的 proximity"的质疑（Table 3），并自我否证细长目标假设（A.12）。

---

## 七、局限与可疑之处

**作者自承的**：Checklist 第 2 项 Limitations 为 **[No]**，即**无单独局限性章节**；第 5 项 Open access 为 **[No]**，代码与 checkpoint 投稿时**未公开**；Theorem 1 只是 in-sample RSS 分析，$\lesssim$ 含未解析确定的低阶松弛项；§5.1 的 bootstrap CI **不测训练种子方差**；A.6 的 RLOO 实验是**单种子兼容性检查**（REINFORCE++/DAPO 留给未来工作）；Table 5 明确 Seeds = 1。

**我认为可疑或没讲清楚的**：

1. **单种子 + 只有测试集 CI 是最硬的伤。** 主表增益下界仅 **+1.0**，单种子下完全可能来自训练噪声，bootstrap CI 又易被误读为误差棒；7B 缩放同样共用单种子。至少要 3 个种子并报种子间方差。
2. **S2 让最常见的 mixed 情形几乎失效。** 1 命中 / 4 失误时，**那条唯一命中的 rollout** 的 LOO 子集是 4 个全 0 的 miss，零方差 → $\hat\rho_d=0$ → $w=0$ → 退回族基线：信息量最大的样本恰用不上几何基线，只有 4 条 miss 受益。而论文**未报告 $w>0$ 的组占比**。
3. **$\hat\rho_d$ 在 4 个点上噪声极大，阈值 0.3/0.7 是手调常数。** A.5 报告二进制奖励下 mixed-hit 组 median $\hat\rho^2_{\text{ens}}$ 仅 **0.18**、median $w=$ **0.0**——即**主实验下趋势支路多数时候根本没开**，主表增益中它的份额因此可疑。
4. **all-miss 支路用了真值 $T(x)$ 的质心与对角线。** 作者称之为外部监督，但与 GUI-G² 的**公平性**存疑：SCA 既加了额外真值几何信息又改了优势层。Table 6 部分处理了这点（SCA+GUI-G² 仍超 GUI-R1+GUI-G² 达 +4.2/+12.7），但**精确质心与高斯软奖励的信息量不等价**。
5. **$k_{\text{prox}}$ 的形式与 $\Delta_d(\mathcal{A})>0$ 假设都缺支撑。** "常数 + 反比"的带宽是纯工程选择，未消融更简单的替代；而若两条 rollout 在同一方向投影上几乎重合，$(L_d/\Delta_d)^2$ 会爆掉、界失效，S1 剔除后若只剩 1 个点对该方向被置零，**等于承认该假设常被违反**。论文也未报告 $\max_d(L_d/\Delta_d)^2$ 的实测分布。
6. **"涌现式课程"（Figure 5）是描述而非解释。** all-hit 从 1% 涨到 65% 是策略变好的**结果**；EGR 从 95% 掉到 30% 也可能只是"后期基线趋同"，论文未设对照组排除这一更简单解释。

---

## 八、与同批论文的关系

> **说明**：本笔记只用 `1008_Spatial_Credit_Assignment.txt`，**未打开其余 9 篇正文**。以下判断依据任务给的线索清单与本篇的 Related Work，涉及他篇内部细节者**需对照原文核实**。

- **22669_GraphGPO（图结构步级信用分配）——最直接的竞争/互补。** SCA 做**组内**信用分配（同一 prompt 的 $N$ 条 rollout 之间按坐标分信用），GraphGPO 若是**步级**信用分配则处理**同一轨迹内跨 step** 的信用，二者维度正交。**自然组合是把 SCA 嵌进步级图结构**：SCA 决定"这一步哪个动作更好"，GraphGPO 决定"哪一步更该被问责"。若它也宣称用几何信息构造基线，需判断是竞争还是互补。
- **15751_HGPO（多层级分组）——承继但有张力。** SCA 的 regime 路由本身就是分组，与本篇**共享同一动机**。**冲突点**：子组变小后 $N\ge4$ 且 $\Delta_d>0$ 的前提可能不满足，**方向集成会直接不可用**。
- **18_PA-MoE（阶段感知专家路由）——互补。** SCA 用**规则式门控**路由到不同信用支路，PA-MoE 提供**可学习路由**。互补方向：**把手门控换成可学习路由**，并用本篇的 EGR 与 $w$ 分布做诊断。
- **35074_KPO（token 级重要性采样比平滑）——正交可叠加。** KPO 动**重要性采样比的平滑**（裁剪/权重侧），SCA 明确"不改裁剪、只改基线"，二者作用于式子的不同位置：**SCA 让优势更有信息量，KPO 让权重更稳**。
- **1406_CITA、1423_CIPO、1432_PhGPO、14071_AgentBrew、13085_AAWM**：若它们也围绕策略优化目标/比率或分组做改进，则 SCA 的卖点——**只换基线、其它全不动**——使它天然是它们的下游插件。**这几篇不做断言。**

**小结**：SCA 是**"信用分配"线上最贴近优势估计器内部的一篇**（GraphGPO 偏步级、HGPO 偏分组、PA-MoE 偏路由、KPO 偏比率），故**更像可插拔组件而非竞品**。

---

## 九、如果我要接着做，可以做什么

**方向 A：把固定 4 方向投影换成数据驱动方向，直接优化 Theorem 1 的代价项。**

- **改什么**：把角度网格换成由组内 rollout 位置 PCA 主轴给出的方向，或令 $\alpha_d\propto\hat\rho_d^2/(1+\lambda(L_d/\Delta_d)^2)$，在权重里惩罚代价项。
- **为什么可能有效**：Theorem 1 的代价是 $\max_d(L_d/\Delta_d)^2$，而现有权重完全没考虑 $\Delta_d$；固定网格里若某方向恰让两条 rollout 投影几乎重合，代价爆掉，但该方向 $\hat\rho_d$ 未必低。
- **第一步（零训练成本）**：在 GUI-ODYSSEY 前 1000 组上记录每个方向的 $\hat\rho_d^2$、$(L_d/\Delta_d)^2$ 与 S1 剔除后剩余点对数，画其分布。**只要 45°/135° 方向有一批该值很大的组，方向 A 就有明确空间**。

**方向 B：去掉对真值目标质心的依赖，让 proximity 变成自监督。**

- **改什么**：Eq. (9) 的 $c(T(x))$ 换成**在线维护的目标中心估计**（如该任务历史命中 rollout 坐标的滑动中位数），$\operatorname{diam}(T(x))$ 换成该估计的组内散布尺度。
- **为什么可能有效**：该支路必须知道 $T(x)$，这限制了 SCA 用在**没有真值目标框、只有稀疏成功信号**的环境，也是它与 reward shaping 的公平性争点。
- **第一步（离线）**：统计每个 all-miss 组"历史命中点中位数"与真值质心的归一化距离误差，并检查前 50 步该估计是否可用。**若不可用，就退化为"滑动窗口 + 组间共享"。**

**方向 C：用置信下界替代手调门控，并补上 $w$ 分布诊断。**

- **改什么**：把 Eq. (8) 的 $w$ 换成基于 $\hat\rho_{\text{ens}}$ **置信下界**的版本：对 $N-1=4$ 个点做 Fisher $z$ 变换取 LCB，再令 $w=\text{LCB}^2$。
- **为什么可能有效**：$\hat\rho$ 在 4 个点上标准误极大，固定阈值 0.3 既可能被噪声伪相关误开、也可能误关；A.5 显示 median $w=0.0$，**说明主实验里门控几乎一直关着**。
- **第一步（零训练成本）**：对每个 mixed-hit 组同时算 LCB 门控与现用 $w$，画**两者随训练步数的分布对比**及"$w$ 由 0 转正"的组占比——**动代码前先知道门控有没有在工作**。

---

## 十、自测问题

1. **[方法内部]** Eq. (3) 说 SCA "故意有偏"。**为什么把基线从 $b(\{r_j\})$ 改成 $\hat R_{-i}(a_i)$ 会破坏无偏性？** Eq. (8) 用什么机制控制偏差、在什么极限下 $B_i$ 消失？
2. **[方法内部]** Theorem 1 右边两项分别是什么？$\Delta_d(\mathcal{A})$ 出现在分母上意味着什么——**什么 rollout 几何配置会使这个界失效？** 论文用哪个 safeguard 应对，代价是什么？
3. **[方法内部]** 为什么 $N=5$ 时不直接做二维仿射回归？请从**参数量/留一后可用点数**与**击穿点**两个角度回答，并说明 $\alpha_d\propto\hat\rho_d^2$ 为何必要。
4. **[方法内部／批判]** S2 说"1 命中 / 4 失误"会自动变成 $w\to0$、退回族基线。请**推导**这一点（提示：那条命中 rollout 的 LOO 子集是什么），并判断：**趋势支路对这唯一一条 hit rollout 是否完全不起作用**？若成立，对"优势层贡献"的主张意味着什么？
5. **[实验设计]** 论文用 bootstrap 95% CI（1000 次 test-set 重采样）报 headline，且每配置只跑**单种子**。**(a)** 该 CI 测什么方差、**不**测什么方差？**(b)** 若训练种子方差与之同量级，**+1.0** 的增益意味着什么？**(c)** 设计最小补充实验判定增益是否真实。

---

*说明：本文所有数字均逐字取自 `_tools/txt/1008_Spatial_Credit_Assignment.txt`。因提取文本按页面绘制顺序输出，表格单元格会竖排（例如 Table 1 的 12 个 SCREENSPOT-PRO 子列与 4 个 SCREENSPOT 子列各占一行），笔记已按表头对齐；如需逐格核对请对照 PDF 表 1。*
