# HarnessPAI 借鉴重评 × Task-level Harness 设计

> 基于 2026-10-13 修正后的事实基线（分辨率归因修正 + round9 部署实测）
> 对照论文：HarnessPAI: An Evolving Harness for Physical AI（arXiv:2609.29166）
> 结论先行：**旧文档里"暂缓"的感知定位原语现在解锁了；"远期"的数据飞轮现在变成最大杠杆；task-level 设计的核心是把 round9 的 stage 表当任务图用，先归因、后写码、只在弱节点介入。**

---

## 1. 事实基线更新（今天改变了三个前提）

| 前提 | 旧认知（10-12 及以前） | 新事实（10-13 实测） | 对设计的影响 |
|---|---|---|---|
| 观测 | policy 侧只有 224 RGB，无深度无分割 | **`_preprocess_obs` 拍平整个 obs 发给 policy**：全分辨率 wrapper 下 `depth_linear` 深度图 + `cam_rel_poses`（相机相对基座位姿）都在 obs 里，harness 合法可读 | 感知定位原语从"工程量最大的 sim 端改造"变成 **policy 侧零注入即可行** |
| VLA 原子能力 | "末端从未接近物体、夹爪从未被阻挡" | 全分辨率下 round9 抓到并放入 2/6 个玩具（q=0.333），stage 推进到 7/14 | HarnessPAI 的硬门槛过了：**是"抓得少/后程乏力"，不是"不会抓"**——落在"代码能救"区间 |
| 旧 harness | 机制跑通、运动指标改善 | **全分辨率配对实验：四版 harness 全部低于裸跑**（servo 0.0 / 旧 harness 0.167 / critic v4 0.0 vs 裸跑 0.333），介入把手臂位移从 40/45 推到 80/83 | HarnessPAI 第 8 条的反面教材已实测：**为"看不见"设计的机制在"看得见"之后全是干扰** |

另外两个新增事实：
- **round9 已全量部署**（`b1k-policy-r9`，单 100-task ckpt），实测与 champion 裸跑打平（301 都是 0.3333）。
- **round9 自带 stage 头**：每个任务的 stage 数在 checkpoint 的 `alignment_2026.json` 里（task 7 = 15 级），推理时输出 stage logits——**免费的任务进度信号，就是 HarnessPAI L1 任务图的运行时探针**。

---

## 2. HarnessPAI 借鉴点重排（优先级在新事实下洗牌）

### 2.1 优先级上升的（现在就该抄）

| 借鉴点 | 旧文档判定 | 新判定 | 理由 |
|---|---|---|---|
| **三类原语平权（#9）** | "高价值缺口，但工程量大" | **P0，且工程量骤降** | 深度+相机位姿已在 obs 里，不用动 sim 端；EefController（雅可比伺服，已实现且验证过误差 0.472→0.024）直接复用做执行器 |
| **能力归因 good/needs-code/mixed（#4）** | "理念可抄，没自动化" | **P0，第一次真的能做** | 以前所有节点都是 0 分没法归因；现在 stage 能推进到 7/14、能抓 2 个，**有了非零基线才能区分强节点和弱节点** |
| **收敛程序采数据 + SFT（#10）** | "远期" | **P0.5，最大杠杆** | 你们本来就有训练闭环（round2→round9 就是续训迭代）！HarnessPAI 这条 +38.8pp 的路径对你们是现成的：harness 辅助的成功轨迹 → round10 训练数据 |
| **Success Contract + 确定性 gate（#2）** | "部分可抄" | **P0，零成本** | 单实例方差 ±0.25（0/0.167/0.333/0.5 都出现过）已两次把我们带进沟里，负例显式化 + 多实例 promote 门是防呆必需品 |

### 2.2 保持原判的

- **Node-keyed 技能库（#3）**：骨架已有（`skill_library.py` L1–L4 已接入），仍然正确——但现在**有条件往里攒第一个真 repair 了**（见 §3.4）。
- **双时间尺度（#1）/代码作接口（#5）**：已天然具备。
- **Writer/Diagnoser 双 LLM、分层候选管理（#6/#7）**：继续缓，纯规则阶段不需要。

---

## 3. Task-level Harness 设计（核心交付）

### 3.1 L1 任务图 = round9 的 stage 表，不用自己发明

HarnessPAI 的 L1 是"每任务一个 workflow graph"。我们手里已经有现成的：

```
alignment_2026.json 的 task_num_stages[task_id]   ←  模型自带，逐任务
stage 头的 logits（15 维）                          ←  推理时免费输出
wrapper 的投票推进（votes_to_promote=2）            ←  已实现
```

**设计**：harness 的 L1 图直接取 `(task_id, num_stages, stage→语义节点映射)`。语义节点映射按 task 写一份小表（如 task 7：stage 0-2 ≈ 导航到桌、3-6 ≈ 逐玩具抓取循环、7-13 ≈ 搬运放置循环、14 ≈ 终态校验），**这是静态任务知识，不碰 privileged 运行时状态，评测合法**。

### 3.2 能力归因协议：先归因，后写码（HarnessPAI 的方法论核心）

对每个任务先跑 1-2 轮 **instrumented 裸跑**（VLA 原样 + 全遥测，零介入），产出逐节点归因表：

| 信号 | 来源 | 回答的问题 |
|---|---|---|
| stage 停留/回退轨迹 | stage 头（已有，policy 日志） | 哪个 stage 是瓶颈（301 实测：大量时间停在 3-4，最高 7） |
| 夹爪闭合并被阻挡 | proprio `gripper_qpos`（中间值=夹住物体，Plan B′ 的 critic 判据直接复用） | grasp 节点成功/空抓率 |
| 逐玩具入箱进度 | q_score 的 per-object 谓词（需加遥测：以 1000 步粒度快照 box 内玩具数） | **2/6 的"2"是哪两个、"4 个没进"卡在哪个节点** |
| 末端轨迹 vs 桌面/箱子位置 | eef_pos（局部系）+ cam_rel_poses | approach/transport 节点的几何质量 |

**产出物**：per-task 的 `good / needs-code / mixed` 节点表。只在 `needs-code` 节点写介入，`good` 节点物理上不许 harness 触碰（写成 assert 级的纪律，不是口头约定——critic v4 的教训）。

以 301 的实测为例，归因假设（待 instrumented run 确认）：
- 前段（stage 0→3）推进正常 → navigate/approach = **good**
- 中段在 3-4 停留最久 → 逐玩具循环里**后几个玩具**的 find/approach = **needs-code**（可能原因：剩下的玩具位置更偏、或 VLA 对已抓过的玩具形成重复吸引）
- 7/14 后再未推进 → 后段 transport/place = **mixed**（能放进去 2 个说明能力有，但未覆盖全部）

### 3.3 介入原语：三类，全部"VLA 优先、节点级兜底"

**A. 感知定位原语（新解锁，HarnessPAI 的 grounding 同构）**

```
head::rgb (720×720) ──颜色/轻量检测──▶ 目标玩具像素质心 (u,v)
head::depth_linear[u,v] ──反投影──▶ 相机系 3D 点
obs["cam_rel_poses"] ──变换──▶ 机器人局部系 3D 点 P_target
EefController（已有）──伺服──▶ pre-grasp 位姿 → 交还 VLA 闭爪
```

- 合法性：三个输入全部来自 policy 本来就收到的 obs，零 privileged 信息。
- 相机内参：`cam_rel_poses` 只给了外参，内参需一次性标定（head 相机 pinhole，fov 固定——在 sim 里一次读出后写死，属于静态知识）。
- 检测器：picking_up_toys 的玩具颜色鲜明，颜色阈值 + 深度一致性即可起步；不需要 SAM。
- **分工对齐 HarnessPAI Table 3**：grounding/approach 用代码，**grasp 闭爪和 place 交还 VLA**——VLA 已经证明会抓会放（2/6 是能力证据）。

**B. 节点级卡死看门狗（保留旧 harness 里唯一被证明无害的部分）**

- 触发条件收紧到**节点语义级**：stage 在同一节点停留 > N 步 **且** 夹爪空抓证据（闭合到底且无阻挡）→ 才允许介入。
- 介入动作 = 调用 A 的瞄准原语**一次**，抓到（夹爪被阻挡）立即交还 VLA。**不做网格扫描、不做姿态强制、不做持续修正**——这三样已被全分辨率配对实验证伪（位移翻倍、得分归零）。
- 预算有界：每节点最多 M 次介入（对齐 HarnessPAI"有界组合测试"+ 我们的"有界占用、用完让位"契约）。

**C. 持握/放置验证（旧 CarryCritic 的新生）**

- 以前"抓到后 40 步释放"修了也没用（因为根本抓不到）；现在真实抓握会发生，这条 critic **第一次有了用武之地**：
  - 闭合被阻挡 → 标记 carrying，交给 stage 头等放置信号；
  - 搬运中滑脱（夹爪宽度回弹）→ 归因 `dropped_in_transport`，触发一次 re-approach；
  - 到 place 节点且夹爪张开且不再有阻挡 → 验证入箱（逐玩具遥测确认）后才算该节点完成。

### 3.4 Success Contract + 技能库落地（HarnessPAI #2 + #3）

**Per-task contract**（写死在 harness 配置里）：

```yaml
task_7_picking_up_toys:
  goal: "6 toys inside toy_box"          # 权威目标（BDDL）
  non_success_signals:                    # 显式负例（HarnessPAI 的"不算成功"清单）
    - gripper_open_and_no_progress_500steps   # 原地空转
    - arm_pose_back_to_init                    # 姿势归位≠完成
    - stage_head_repeated_regression_x3        # stage 头来回抖
  promote_gate: "q>0 on >=5 instances AND mean_lift_over_bare >= 0.05"   # 多实例全成才入库
```

**技能库 L3 键** = `(task_id, node, problem_label)`，problem_label 由 critic 归因产生（`empty_grasp` / `dropped_in_transport` / `stuck_at_node`）。第一个可攒的 repair 就是 3.3-A+B 组合："stuck_at_node + empty_grasp → 单次瞄准介入 → 交还"。攒够 2-3 个任务后，跨任务的 node 级聚合才开始有意义（HarnessPAI 的"Aha 时刻从多轮降到 1-2 轮"的前提）。

### 3.5 数据飞轮：harness 收敛轨迹 → round10（对你们是现成闭环）

这是本次重评**最大的优先级变化**。HarnessPAI §4.5/6.3：收敛程序开环跑出的成功轨迹（带 success gate 硬信号）直接作 SFT 数据，π0.5-LIBERO **+38.8pp**，零人工标注。

你们的现实优势：**训练管线是自己的**（vlatmp_0924 数据集 + round2→round9 续训历史）。映射过来：

```
round9 裸跑（0.333）→ +节点级介入收敛（目标 0.6+）→ 成功轨迹（6/6 入箱）
     ↓ 过滤：仅保留 success gate 硬通过的完整轨迹
round10 训练数据（合法、免标注、分布内）→ 续训 → 弱节点能力内化
     → 下一轮归因时该节点变 good → harness 介入范围收缩
```

这同时回答了"harness 和 RFT 是否二选一"：**不是，是同一飞轮的两级**（HarnessPAI 原文立场），而且你们刚好两个都有。

---

## 4. 实验纪律（防止第四次踩同一类坑）

1. **永远不用 224 wrapper 跑任何对照**（已在 `run_sim*.sh` 固化默认值）。
2. **单实例单 rollout 不下结论**：配对设计，public_test 索引 0..9（全局 301..310）串行跑 bare 与 +harness 两组，比较均值；单实例方差 ±0.25，10 实例配对可把检出阈值压到 ~0.1。
3. **每开一个机制跑一次归因 run**：先 instrumented 裸跑确认"这个节点的确弱、且是这个 problem"，再写 repair；禁止全家桶默认开启（critic v4 教训：`POSTURE/WSCAN/SEARCH` 默认值全是 1，一次实测就现形）。
4. **promote 前过 contract gate**：q>0 不算数，要 ≥5 实例 + 均值提升 ≥0.05。

## 5. 落地顺序建议

| 序 | 动作 | 依赖 | 预估 |
|---|---|---|---|
| 1 | **Instrumented 裸跑**：加逐玩具入箱遥测（1000 步粒度）+ 夹爪阻挡事件日志，301-303 跑 3 实例 | 只改 policy 侧 wrapper（我们已有 r9 代码挂载点） | 半天改 + 1.5h 跑 |
| 2 | **产出 task 7 归因表**（哪两个玩具进了、四个卡在什么节点/problem） | 序 1 | 数据分析，1h |
| 3 | **相机内参标定** + 感知定位原语 v0（颜色阈值 + 深度反投影 + EefController 伺服） | 序 2 确认 needs-code 节点 | 1-2 天 |
| 4 | **节点级看门狗 + 单次瞄准介入** + contract/promote gate | 序 3 | 1 天 |
| 5 | **配对评测**：bare vs +harness × 10 实例 | 序 4 | ~10h 后台串行 |
| 6 | （达标后）**成功轨迹导出 → round10 数据** | 序 5 promote | 你们训练侧接管 |

## 6. 一句话总结

HarnessPAI 最值得抄的从来不是它的演化循环，而是三条纪律：**先归因再写码、只在弱节点介入、成功要过确定性 gate**。今天分辨率修正 + round9 打平基线之后，"归因"第一次有了非零的对照面，"感知原语"第一次有了合法的输入（深度+相机位姿就在 obs 里），而"数据飞轮"第一次有了你们的训练闭环可以接——task-level harness 的正确定位是：**给 round9 补最后 4 个玩具，然后把补过的经验喂给 round10。**
