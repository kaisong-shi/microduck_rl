# 22-鸭足球（11v11）任务列表

> 所有数字均在本机实测（RTX 4090 24 GiB, driver 580.167.08, torch 2.9.1+cu128, warp 1.12.0），不是估算。
> 上游方案讨论见对话记录；本文件是唯一的执行清单，改设计先改这里。环境快照见 `docs/soccer_env.md`。

---

## 进度总览（2026-10-29）

> **P0–P4 全部完成，P5 完成 S0（1v1）并训出进球。S1–S5 待跑。**
> 测试 **391 passed / 1 skipped**；`soccer/` 9 个模块 **2,694 行**，配套测试 **1,730 行**。

### 阶段完成度

| 阶段 | 内容 | 状态 | 验收证据 |
|---|---|---|---|
| **P0** | conda `microduck_rl` = py3.12 + CUDA | ✅ | 5 门全绿（G1–G5），`docs/soccer_env.md` |
| **P1** | 场地 7.0×4.5 + 球（滚动阻力） | ✅ | 27 条测试；球 1 m/s → 1.88 m / 3.76 s，无蠕动 |
| **P2** | 22 鸭世界（allcollisions） | ✅ | 15 条测试；`nq=469 nu=308`，2000 步无 overflow/NaN |
| **P3** | 97D 观测 / 奖励 / 事件 | ✅ | 47 条测试；180° 旋转 + 队别互换后观测逐元素相同 |
| **P4** | `SoccerEnv` + `E*22` lane + PPO | ✅ | 9 条测试；PPO 实跑通（含动作路由测试） |
| **P5-S0** | 1v1 课程 | ✅ | 触球率 100 %、球位移 7.3 m/回合、进球峰值 0.051/回合 |
| **P5-S1..S5** | 2v2 → 11v11 | ⬜ | 见 §7 |
| **P6** | 无头评测 | 🟡 | `scripts/soccer_eval.py` 已能出表；CSV/对手池未做 |
| **P7** | viewer / 录像 | ⬜ | — |

### 代码地图

| 文件 | 行数 | 职责 |
|---|---|---|
| `soccer/pitch.py` | 422 | 场地/球门/围栏几何、球的滚动阻力、**全部尺寸常量** |
| `soccer/scene.py` | 311 | 22 个独立 Entity、4-4-2 与小场阵型、qpos/ctrl 切片、`verify_world` |
| `soccer/obs.py` | 339 | 每鸭 97D 观测（体坐标系 + 队相对世界量） |
| `soccer/rewards.py` | 333 | 进球 / 势差 / 走向球 / 触球 / 扎堆 / 倒地 / 僵局（全 NaN-safe） |
| `soccer/events.py` | 121 | 中圈圆盘球位、逐鸭抖动、开球球权 |
| `soccer/env.py` | 555 | `SoccerEnv`：自持 step 循环的 22-agent 环境 |
| `soccer/vecenv.py` | 98 | `E*22` lane 的 rsl_rl 适配器（唯一做 lane↔world 重排的地方） |
| `soccer/rl_cfg.py` | 82 | PPO 超参（γ=0.995、lr=3e-4、entropy=0.003） |
| `scripts/train_soccer.py` | 126 | 训练入口（`--n-team` / `--num-worlds` / `--entropy-coef`） |
| `scripts/soccer_eval.py` | 169 | 无头评测（P6-lite，JSON 输出） |
| `scripts/sync_conda.sh` | 125 | conda 环境重建（幂等 1.4 s） |

### 训练进度（实测数字）

| 项 | 值 |
|---|---|
| 11v11：22 鸭 × 512 世界 = **11264 lane** | 409 ms/控制步 → **9.8 s/迭代**，显存 12.6 GB |
| 1v1：2 鸭 × 2048 世界 = **4096 lane** | **1.0 s/迭代** |
| S0 训练量 | 1200 迭代 ≈ 28 min |

S0 每 300 迭代（96 回合评测）：

| iter | 进球/回合 | 球位移/回合 | 触球/回合 | 倒地时长 | 僵局结束 |
|---|---|---|---|---|---|
| 300 | 0.020 | 4.26 m | 6.49 | 0.184 | 38.8 % |
| 600 | 0.042 | 4.79 m | 6.43 | 0.368 | 35.4 % |
| **900** | **0.051** | 5.22 m | 7.40 | 0.294 | 24.5 % |
| 1199 | **0.010** | **7.28 m** | 8.33 | 0.393 | 18.6 % |

### 当前未解决（S1 之前必须先处理前两条）

| # | 问题 | 数据 | 判断 |
|---|---|---|---|
| 1 | **戳球农场** | 进球 0.051(900) → 0.010(1199)，触球 7.4→8.3 | `touch`(0.15) + `ball_approach` 为"乱捅"付钱，进球太难 |
| 2 | 倒地 / 趴着走 | 平均倾斜 **53°**，倒地 39 % | 是第 1 条的副产品，**先别加重惩罚** |
| 3 | `action std` 漂移 | 1.0 → 2.10 | 撞 ±1 clamp 后 log-prob 不一致的风险 |
| 4 | 僵局仍 19 % | 球可能卡墙角/门柱 | 僵局判定可加"贴边"条件 |
| 5 | P6/P7 | — | 评测表已有，CSV/对手池/viewer/录像未做 |

**下一步**（建议顺序，已与用户确认"先修奖励再进 S1"）：
1. `touch` 只在球被朝进攻方向推动时计入（或权重 0.5 → 0.2）；
2. `ball_goal_progress` 权重 15 → 25–30；
3. 从 **model_900** 续训 S0 300–400 迭代，确认进球**持续上涨**而非再次跑偏；
4. 进 S1（2v2，从 S0 最好 checkpoint 暖启）→ S2 4v4 → S3 6v6 → S4 11v11 → P6 门禁 → P7 展示。

---

## 0. 已冻结的设计决定

| 项 | 决定 | 依据 |
|---|---|---|
| 机器人数量 | 22（11 v 11） | 用户要求 |
| 场地 | **7.0 × 4.5 m**（x 为长边），中圈 r=0.61，禁区 1.10 × 2.67 | 密度 4.8 身高/鸭（人类 10，SPL 5.8）；射程/场长 0.28 |
| 球门 | **0.70 × 0.35 m**，深 0.25，20 mm 方柱 | 占场宽 15.6 %（人类 10.8 %）；高 = 1.4 身高（人类 1.36） |
| 球 | 70 mm / 15 g，`ball.xml` **原样不动**；只在内存 spec 上给 freejoint 加 `frictionloss=2.15e-4` | 见 §3 |
| 碰撞模型 | **`robot_allcollisions.xml`**，训练与评测同模型 | 用户决定 |
| 围栏 | 边界外 0.30 m，不可见，高 0.20 m | 挡住球和鸭，避免无限出界重置 |
| 物理 | `timestep=0.005`，`decimation=4`（控制 20 ms / 50 Hz） | 与仓库其它 env 一致 |
| `njmax` / `nconmax` | **2600 / 1100** | 22 鸭成堆实测 nefc 2017（原估 1850 不够） |
| 观测 / 动作 | 每鸭 **97D** / **14D**；critic = 97D + 全局附加块 | 见 §5.1 |
| 训练范式 | 参数共享 **IPPO**（每鸭一条 lane），同质自博弈 | 见 §6 |
| 场地坐标 | 每鸭观测在**规范坐标系**（我方进攻方向恒为 +x） | 消掉队别 one-hot，学习难度减半 |
| 赛制 | 先到 3 球 或 3 分钟 | 横穿 41 s |
| 训练 episode | 30 s（S0–S3）/ 40 s（S4） | 一个控球回合 ≈ 13 s（踢 1.9 m + 停 3.8 s + 追上 11 s） |
| 部署 | **不做**（纯仿真）。不导出 ONNX，不受 61D 契约约束 | 用户决定 |

### 0.1 关键实测数字（写死在测试里）

| 量 | 值 |
|---|---|
| 22×512 世界（11264 lane）控制步 | **409 ms**（含观测/奖励）→ 迭代 **9.8 s** |
| 22×512 世界显存 | **12.6 GB**（本体）/ 19.2 GB 全卡 |
| 22×128 | 159 ms（1.24 ms/世界，比 512 差一倍） |
| 22×512, nconmax=1600 | **OOM**（CUDA graph 创建） |
| `max nefc/world` 实测 | **2017**（随机动作 + 倒地成堆） |
| 世界规模 | `ngeom=3134  nq=469  nv=446  nu=308  nbody=335  njnt=331` |
| 22 鸭构建+编译 | **2.2 s** |
| 球 1 m/s 停止距离 / 时间 | **1.88 m / 3.76 s**（`frictionloss=2.15e-4`，真实 15 g 球、纯滚动） |
| 球 0.3 m/s（推球） | 0.17 m / 1.1 s |
| 走速 / 转向 | 0.17 m/s / 0.45 rad/s |
| 横穿全场 | 41 s |

---

## 1. 里程碑总览

| 阶段 | 产出 | 验收 | 预算 |
|---|---|---|---|
| **P0** | conda `microduck_rl` = py3.12 + CUDA 栈 | 5 个验收门全绿 | ✅ 完成 |
| **P1** | `soccer/pitch.py` 场地+球 | 球滚动标定测试绿 | ✅ 完成 |
| **P2** | `soccer/scene.py` 22 鸭世界 | 1000 步无 overflow/NaN | ✅ 完成 |
| **P3** | 观测 + 奖励 + 事件 | 形状/符号/量程测试绿 | ✅ 完成 |
| **P4** | 环境 + lane 包装 + PPO | rsl_rl 跑通 | ✅ 完成 |
| **P5** | S0 1v1 → S4 11v11 课程 | §8 指标 | S0 ✅（14 min）；S1–S5 **20–30 h GPU** |
| **P6** | 无头批量评测 + CSV | 50 局统计 | 0.5 d |
| **P7** | viewer 回放 / 录像 | 可交付 | 0.5 d |

**止损点（已执行）**：P3 结束后先在 1v1 上跑 300 迭代 —— 第一次实测球位移 0.27 m、触球率 10 %，触发了止损，回头查出 `action_scale`（0.25→1.0）、`spawn_poses`（漏传 `n_team`）、缺失 `ball_approach` 三个问题后才达标。详见 §7.2。

---

## 2. P0 — conda 环境 ✅ 完成（2026-10-29）

**工具**：`scripts/sync_conda.sh`（以 `uv.lock` 为唯一真源导出 requirements 再 `uv pip install`；不能直接 `pip install -e .`）
**快照与验收输出**：`docs/soccer_env.md`

- [x] **P0-T1** 写 `scripts/sync_conda.sh`
  - 处理三个坑：`uv export` 首行 `-e .` 必须落在仓库根；bam 的 `protobuf<4` / `onnx` 降级 / `zmq==0.0.0` 坏 wheel 由 `[tool.uv] override-dependencies` 解决，只有 `uv` 认；不装 conda 的 cudatoolkit/cudnn/pytorch。
- [x] **P0-T2** 删除旧 env（py3.10、无 torch）并重建为 py3.12（→ 3.12.16）
- [x] **P0-T3** 安装全部依赖（154 包，含 git bam `62bd8ce1`）
  - 关键修复：导出的 `-e .` 会让 `uv pip` 撞上 bam 的 URL 冲突（项目元数据写 **branch** `mjlab_frictionloss`，锁文件写 **commit** `62bd8ce1`，指同一物但 uv pip 拒绝）。**对策**：导出时滤掉 `-e .`，装完依赖再 `uv pip install --no-deps -e .`。
- [x] **P0-T4** 验收门 G1：`3.12.16 / 2.9.1+cu128 / cuda 12.8 / available True / device_count 1` ✅
- [x] **P0-T5** 验收门 G2：`wp.get_device('cuda') → cuda:0`（RTX 4090 sm_89）✅
- [x] **P0-T6** 验收门 G3：`bin/train` 存在，注册了 `Mjlab-Velocity-Flat-MicroDuck` / `Mjlab-BallKick-Flat-MicroDuck` 等 ✅
- [x] **P0-T7** 验收门 G4：5 迭代跑完，0.51 s/迭代，`nan_state=0`，`Episode_Reward/*` 正常 ✅
- [x] **P0-T8** 验收门 G5：`bam.model` / `mujoco 3.10.0` / `mujoco_warp 3.8.1` 导入正常 ✅
- [x] **P0-T9** `requirements-conda.txt`（及 `MUJOCO_LOG.TXT`）加入 `.gitignore` ✅
- [x] **P0-T10** 记录 env 快照到 `docs/soccer_env.md` ✅
- [x] **P0-T11** 幂等性验证：无变更时重跑 `sync_conda.sh` 用时 **1.4 s**，零安装 ✅

> **已知坑**：conda 的 `libstdc++` 可能过旧 → `GLIBCXX_3.4.30 not found`；对策 `conda install -c conda-forge "libstdcxx-ng>=13"`。当前 `libstdcxx 15.2.0`，未触发。

---

## 3. P1 — 场地与球 ✅ 完成（2026-10-29）

**产出**：`src/mjlab_microduck/soccer/pitch.py`（+ `soccer/__init__.py`），测试 30 条全绿（4.6 s）。
渲染验证：`/tmp/opencode/pitch_view.png`（绿茵 + 白线 + 中圈 + 禁区 + 两端球门 + 中圈球）。

- [x] **P1-T1** 参数化生成：草地平面（`include_floor=True` 的自立版）、边线/中线/中圈/禁区（**纯视觉**，`contype=conaffinity=0`）、2 个球门（20 mm 柱 + 横梁 + 实心网箱）、4 面不可见围栏（`rgba` alpha=0 但可碰撞）✅
- [x] **P1-T2** 球 EntityCfg：`ball_spec()` 内 `MjSpec.from_file(ball.xml)` → `sp.joint("ball_free").frictionloss = 2.15e-4`，**文件不动、condim/friction 不动** ✅
- [x] **P1-T3** 常量集中在 `pitch.py` 头（`PITCH_L/W`, `GOAL_W/H/D`, `WALL_H/MARGIN`, `BALL_ROLL_LOSS`, `SIM_TIMESTEP`, `CONTROL_DECIMATION`, `NJMAX/NCONMAX`）✅
- [x] **P1-T4** 部分移交 P2：P1 交付 `pitch_spec(include_floor)` / `ball_spec()` / `PITCH_ENTITY_CFG` / `BALL_ENTITY_CFG`；**22 鸭装配在 `soccer/scene.py`（P2-T1）** — 鸭的摆位与 P2 的 `nu/nq` 断言必须一起改，拆开反而容易漂移
- [x] **P1-T5** `tests/test_pitch_geometry.py`（15 条）：尺寸、球门口径/柱位/横梁跨度、围栏内表面位置、球门不出围栏、线不可碰撞、围栏不可见但实体、中圈分段在圆上 ✅
- [x] **P1-T6** `tests/test_ball_roll.py`（9 条）：1 m/s → **1.95 ± 0.10 m / 3.9 ± 0.5 s**；0.3 m/s → 0.18 ± 0.02 m；停住后 10 s 位移 < 1 mm；`d ∝ v²` 的三点校验；**射门端到端**（2 m 外 1.2 m/s，3 个瞄准点：过线、不进网、留在门里、静止）✅
- [x] **P1-T7** `tests/test_ball_no_contact_override.py`（6 条）：编译后 `geom_condim[ball]==3`、`geom_friction[ball]` 与 `from_file` 逐位一致、`body_mass==0.015`、`body_inertia` 一致、唯一差异是 `dof_frictionloss`；且 `ball.xml` 磁盘内容未被改 ✅

**球滚动标定（真实 15 g / 70 mm 球，dt=0.005，初始条件 = 纯滚动）**

| frictionloss | 0.2 m/s | 0.3 | 0.6 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| 1.0e-4 | 0.168 / 1.67 s | 0.378 / 2.51 | 1.513 / 5.04 | 4.204 / 8.41 | 16.82 / 16.85 |
| 1.5e-4 | 0.112 / 1.11 | 0.252 / 1.68 | 1.008 / 3.36 | 2.802 / 5.61 | 11.21 / 11.23 |
| 2.0e-4 | 0.084 / 0.84 | 0.189 / 1.26 | 0.756 / 2.52 | 2.101 / 4.21 | 8.408 / 8.42 |
| **2.15e-4 ✅** | **0.075 / 0.75** | **0.168 / 1.12** | **0.675 / 2.25** | **1.876 / 3.76** | **7.506 / 7.52** |
| 3.0e-4 | 0.056 / 0.56 | 0.126 / 0.84 | 0.504 / 1.68 | 1.400 / 2.81 | 5.604 / 5.62 |
| 4.0e-4 | 0.042 / 0.43 | 0.094 / 0.64 | 0.377 / 1.26 | 1.050 / 2.11 | 4.202 / 4.21 |

规律：`d = 1.95·v²`，`t = 3.9·v`（2.15e-4）；停后 10 s 漂移 **1e-5 m 量级**（无蠕动尾巴）。
**不要用 `condim=6` + `friction[2]`**：那是力矩上限约束，低速留几十秒蠕动尾巴（5e-4 时 1 m/s 要 76 s 才停）。

### ⚠️ 本轮修掉的两个标定错误（写下来防止再犯）

1. **错误的初始自转方向 → 球在打滑而不是滚动。** 沿 +x 前进时无滑滚动的角速度是 **`ω_y = +v/r`**（我最初写了 `-v/r`）。打滑的球被接触摩擦猛拽，1 s 内从 1 m/s 掉到 0.19 m/s，然后拖出十几秒的"尾巴"——看起来像"滚动阻力"，其实完全是另一套物理。
2. **标定用的球不是真球。** 早期探针里手动拼的球没有 `<inertial>`，MuJoCo 按密度算出 **0.1796 kg（真球 15 g 的 12 倍）**。阻力 ÷ 质量 差了 12 倍，任何 `frictionloss` 数值都失去意义。
   → **必须用 `MjSpec.from_file(ball.xml)` 的真实资产**，并且测试里断言 `body_mass == 0.015`。

这两条合起来，把常量从错误的 `4e-4` 改到 **`2.15e-4`**（距离目标 1.95 m 不变，但停止时间从"19 s"变成正确的 3.9 s —— 后者才是 `d = v²/2a` 应有的结果）。

### ⚠️ 网箱为什么是"实心"的（推翻 P1 初稿的 `contype=0`）

MuJoCo 判定一对 geom 是否碰撞是 `(contype₁ & conaffinity₂) | (contype₂ & conaffinity₁)`。球和鸭的 `contype` 都是 1，所以"挡住鸭、放球过去"在单一碰撞类下**做不到**（要么都挡、要么都过）。
改成实心反而更好：1 m/s 的球撞上网后（默认 `solref` 无回弹）**立刻停住并留在门里**，进球事件不用赌"球恰好还在门内"的那一帧。`test_ball_roll.py::test_shot_on_goal_ends_up_inside_the_goal` 锁死了这个行为。

---

## 4. P2 — 22 鸭世界 ✅ 世界部分完成（2026-10-29）

**产出**：`src/mjlab_microduck/soccer/scene.py`；`tests/test_soccer_world.py`（15 条全绿，含 2 条 GPU 用例）。
**未做**：`tasks/microduck_soccer_env_cfg.py`（env cfg 与 P3 的 obs/reward/event、P4 的 lane 包装强耦合，放到 P4 一起落地，避免写两遍）。

- [x] **P2-T1** 22 只 `robot_allcollisions`，4-4-2 镜像摆位（team1 = **180° 旋转** `(x,y,yaw)→(-x,-y,yaw+π)`）。
  **关键设计修正**：不是"一个 spec 塞 22 只鸭"，而是 **22 个独立 mjlab `Entity`**（`d00`…`d21`）。因为 mjlab 的 `Entity` = 一个 articulation：它取 spec 的第一个 joint 当浮动基座、其余 joint 全当铰链——塞 22 只鸭会让另外 21 个 freejoint 被当成 hinge，静默毁掉整个模型。
- [x] **P2-T2** duck-major 顺序断言：`nq=469`（22×21+7）、`nv=446`、`nu=308`、`njnt=331`；`joint(i*15)` 必是 `d{i:02d}/trunk_base_freejoint`；每个 actuator 的 target 都属于本鸭且 14 个互不相同。
- [x] **P2-T3** `NJMAX=2600`、`NCONMAX=1100`、`timestep=0.005`、`decimation=4`（见下方实测，**原定的 1850/900 不够**）。
- [x] **P2-T4** 不注册接触传感器（22 鸭会撑爆 `contact_sensor_maxmatch`）；倒地判定留给 P3，用 `trunk_z` + 投影重力。
- [x] **P2-T5** `expand_bam_friction_fields` 启动事件：留到 env cfg（P4）。**但已实测**：`sim.expand_model_fields(("dof_frictionloss","dof_damping"))` 之后 `dof_frictionloss` 形状 `(nworld, 446)`、`stride(0)=446`（真正逐世界），124 个测试断言了这一点。
- [x] **P2-T6** 308 个执行器覆盖 22×14；断言 **没有任何 actuator 的 target dof 落在球上**（球的 6 个 dof 与舵机 dof 完全不相交 → 关节摩擦 DR 永远碰不到球的滚动阻力）。
- [x] **P2-T7** `tests/test_soccer_world.py`：编译、布局、`qpos0`=阵型、HOME 位姿、切片覆盖与互斥、GPU 250 步无 NaN/发散。
  **长时验证**：另跑 512 世界 × 2000 物理步，无 `nefc/ncon` overflow、无 NaN。
- [x] **P2-T8** 球的 `2.15e-4` 在 `expand_model_fields` 之后逐世界保持（写进 `test_soccer_world.py::test_world_steps_without_overflow_or_nan`）。
- [x] **P2-T9** `test_bam_never_touches_the_ball_dofs`：actuator 目标 dof 集合 ∩ 球的 dof 集合 = ∅。

### 实测数字（写死在测试里）

| 量 | 值 |
|---|---|
| 世界规模 | `ngeom=3134  nq=469  nv=446  nu=308  nbody=335  njnt=331` |
| 构建+编译（22 只鸭） | **2.2 s**（22 次 `MjSpec.from_file`） |
| `max nefc/world`（随机动作 + 成堆倒地） | **2017** → `NJMAX=2600` |
| `nconmax` 需求 | ≥821 且随时间缓慢增长 → `NCONMAX=1100` |
| 显存（512 世界） | **19.2 GiB 总占用**（含本机已占 6.6 GiB）→ 本体 **12.6 GiB** |
| 吞吐 512 世界 | **331 ms / 控制步** = 0.65 ms/世界 |
| 吞吐 128 世界 | 159 ms / 控制步 = 1.24 ms/世界（所以 512 更划算） |
| 单迭代（24 控制步） | 24 × 331 ms ≈ **7.9 s 纯物理** |

### ⚠️ 两个必须记住的坑

1. **`qpos0` 决定 `Simulation.__init__` 能否活下来。** mjlab 建完 `Scene` 后立刻 `Simulation(...)` → `mj_forward`，而它用的是编译后模型的默认状态 `qpos0`。freejoint 的 `qpos0` 来自 **spec 里 root body 的 pos/quat**：22 只鸭都留在原点时，`mj_forward` 面对 22 只互相穿透的鸭 → 约 8 万个接触 → `mj_stackAlloc: stack overflow`（**不是** NaN，是直接崩）。
   对策：`duck_spec(i)` 里把 `trunk_base` 的 pos/quat 设成该鸭的出生位姿。`test_qpos0_is_the_formation_not_the_origin` 锁死。
   注意 `qpos0` 里**关节仍是 0**（HOME 位姿在合并的 `init_state` keyframe 里，由 reset 写入）——这是对的，别"顺手"把它改掉。
2. **`nconmax` 才是显存杀手**，不是 `njmax`。512 世界下 `nconmax=1600` 直接 CUDA graph OOM，而 `njmax` 从 2200 提到 2400 几乎不影响。

---

## 5. P3 — 观测 / 奖励 / 事件 ✅ 完成（2026-10-29，critic 详见下）

**产出**：`soccer/obs.py`、`soccer/rewards.py`、`soccer/events.py`；
`tests/test_soccer_obs.py`(17) + `test_soccer_rewards.py`(22) + `test_soccer_events.py`(8) 全绿。全套 367 条通过。

### 5.1 观测（每鸭 97D，`obs.py::_BLOCKS` 是唯一真源）

**坐标系决定（其余全部由此推出）**：所有**相对量**都在鸭子自己的**体坐标系**
（`v_b = Rᵀ v_world`，完整旋转，不只 yaw）；只有"我在场上哪儿"这类**世界量**乘
`team_sign`（队 0 = +1，队 1 = −1）。

代价与收益都写清楚：鸭子因此**不知道自己的绝对朝向**（体坐标系里朝向恒为 +x），
任何需要绝对方向的东西必须以相对向量给出（"我方球门"块就是为此存在）。换来的是：
**世界旋转 180° + 队别互换后，每只鸭的 97 维观测逐元素相同** —— 不需要队伍 one-hot、不需要镜像表，
学习难度减半。`test_observation_is_invariant_to_a_180_rotation_plus_team_swap` 锁死这条。

| 块 | 维 | 内容 |
|---|---|---|
| joint_pos_rel | 14 | 相对 HOME 的关节位置 |
| joint_vel_rel | 14 | |
| base_ang_vel_b | 3 | 体坐标系 |
| projected_gravity_b | 3 | 重力方向 = 姿态 |
| last_action | 14 | 上一步动作 |
| ball_rel_pos_b | 3 | |
| ball_rel_vel_b | 3 | 相对鸭子的速度（不是地面速度） |
| ball_distance | 1 | 归一化 8 m |
| 最近 3 队友 | 12 | 相对位置 2 + 相对速度 2，各 ×3 |
| 最近 3 对手 | 12 | 同上 |
| 队友/对手质心 | 4 | 各 2 |
| 自身位置 | 2 | 规范坐标归一化 |
| 自身高度 | 1 | trunk z / 0.25 m |
| 角色 | 4 | one-hot GK/DEF/MID/FWD |
| 剩余时间 | 1 | |
| 净胜球 | 1 | **队相对**（own − opp） |
| 上次触球 | 2 | **队相对** one-hot(我方/对方) |
| 我方球门 | 3 | 相对位置 |
| **合计** | **97** | |

> 修正：本文件初版的表格加起来只有 95 维（原稿把自身位置/球门写成 2 维）。现在按上表实现，97 由 `_BLOCKS` 求和并 `assert`。
> 小规模阶段（S0–S3）队伍变小：缺失球员的槽位**恒为 0**（距离 = inf，永不被选中），**布局维度不变**，热启动才不会崩。

- [x] **P3-T1** `build_obs(...) -> (E, 22, 97)` 纯 torch 函数（无 simulator 对象，便于以后换真实感知）✅
- [x] **P3-T2** critic 全局观测 189D —— **改设计**：见下"critic 决定"
- [x] **P3-T3** 对称性测试并入 `tests/test_soccer_obs.py`（含小规模世界）✅
- [x] **P3-T4** 形状/无 NaN/量程/最近邻正确性/小块对齐/球在体坐标系 ✅
- [x] **P3-T5** 全部 penalty 自否定（返回 ≤ 0）+ **正**权重；`assert_penalty_signs()` 供 env 每步自检 ✅
- [x] **P3-T6** `tests/test_soccer_rewards.py::test_every_penalty_is_non_positive_on_a_random_state` ✅
- [x] **P3-T7** 事件：`events.py`（中圈 ±0.3 m 圆盘采样、逐鸭 ±5 cm 抖动、开球球权判定、场内 clamp）✅
  未接进 mjlab `EventTermCfg` —— 与 env cfg 一起在 P4 落地。

**critic 决定（推翻初稿的 189D）**：初稿的 189D 是"全局上帝视角"。但 IPPO 的价值函数只需要
**比 actor 更全**的信息，不需要冗余的全局编码：改成 **critic 观测 = actor 97D + 全局附加 24D**
（球的世界坐标 3 + 球速 3 + 22 只鸭的 (x,y,vx,vy) 由 4 个全局聚合代替）。实现上就是
`build_obs(...)` 的结果再 `cat` 一个 `(E,22,K)` 的全局块 —— 维度在 P4 定，**actor 侧 97 不变**。

### 5.2 奖励（`rewards.py`，权重是 `WEIGHTS` 字典）

| 项 | 形式 | 权重 | 备注 |
|---|---|---|---|
| 进球 | 上升沿事件 ±1，**全员**，**队相对** | +50 | 球在门里就停住（实心网），事件不会重复触发 |
| 球-球门势能 | `Φ' − Φ`，`Φ = −d/8`（**不折扣**） | +15 | 见下"为什么不用 γΦ'−Φ" |
| 触球 | 首次触球 +0.3，5 s 冷却 | +2 | 连续带球 = 每 5 s 记一次 |
| 队友间距 < 0.25 m | `−Σmax(0, 0.25−d)`（返回 ≤ 0） | +1 | 对手靠近**不罚**（那正是对抗） |
| 倒地（倾斜 > 60°） | `−clamp(g_z − (−cos60°))/…`（返回 ≤ 0） | +2 | 0→1 有界，仰面 = −1 |
| 无进展（5 s 球位移 < 0.15 m 且无触球） | 双方 −1（返回 ≤ 0） | +0.5 | 僵局是双方的失败 |
| action_rate / joint_torque / dof_limits | 复用 velocity 配方 | 各 ≤ 0 | P4 接 |

**为什么不用 `γΦ' − Φ`（重要）**：折扣形式会**每步付 `d·(1−γ)`**，
即球离对方球门**越远付得越多** → 策略学会把球留在自己半场。所以用**不折扣的势差**
`Φ' − Φ`：球不动 = 恰好 0（不可farm），前进 = 正，后退 = 负，去而复返 = 抵消。
`test_progress_cannot_be_farmed_by_parking_far_from_the_goal` 与 `test_progress_round_trip_cancels` 锁死。

### ⚠️ 本轮被测试抓住的 4 个 bug（都是"看起来对"的那种）

1. **前向运动学式子维度写错**：`torch.einsum("ij,enj->eni")` 把"球员"和"坐标"两个轴搞混（`pos_xy` 是 `(E,N,2)`，中间轴才是球员）。→ centroid 全错。
2. **四元数分量顺序搞错**：绕 z 转 180° 的合成 `qz⊗q = (−z,−y,x,w)`，我写成按 `q[:,2],q[:,1],q[:,0],q[:,3]` 取——那是 (w,x,y,z) 的第 3/2/1/4 个分量，结果变成**绕 y 转**。用"单位四元数"验证时完全看不出来（退化案例）。
3. **净胜球/上次触球是世界标量直接广播**：两队读到同一个符号 → 破坏了 180° 对称性，也让"我领先吗"变成噪声。改成**队相对**。
4. **`fallen_penalty` 的 clamp 方向反了**：`onset − g_z` 应为 `g_z − onset`，否则站直时付满额、躺下付 0（正好反了）。测试的 `parametrize`（0/30/59/61/90°）一眼抓住。

**教训**：这几类错误在 sim 里都不会报错，只会让训练"怪怪的"。凡是**坐标系/符号**相关的
纯函数，必须先写对称性测试再写实现。

---

## 6. P4 — 环境与多智能体包装 ✅ 完成（2026-10-29）

**产出**：`soccer/env.py`（`SoccerEnv`）、`soccer/vecenv.py`（`SoccerVecEnv`）、`soccer/rl_cfg.py`、
`scripts/train_soccer.py`；`tests/test_soccer_env.py`(9 条)。**PPO 已实跑通**（8 世界 × 3 迭代 + 512 世界测速）。

### 6.1 为什么不走 `ManagerBasedRlEnv`（推翻 P4 初稿）

mjlab 的 manager 层是"**每世界一个机器人**"的结构：一个 action term、一个 observation group、
奖励向量形状 `(E,)`。足球要的是 22 个 agent、每 agent 一份奖励、每鸭 97 维观测。硬塞进去要写
~300 个 observation term + 22 个 action term，而 **reward manager 仍然无法返回 (E,22)**。

所以 `SoccerEnv` 自己拥有 step 循环，只用 mjlab 的下层（`Scene` + `Simulation` + 每实体 actuator）：

| 保留 | 说明 |
|---|---|
| BAM 执行器 | 直接 `duck.set_joint_position_target()` / `duck.write_data_to_sim()` / `duck.update(dt)` |
| CUDA graph 仿真 | `Simulation` |
| 逐世界字段展开 | `sim.expand_model_fields(("dof_frictionloss","dof_damping"))` —— 就是 `expand_bam_friction_fields` 事件做的事 |
| 实体 API | `write_root_state_to_sim` / `write_joint_state_to_sim` / `reset` |

**一步控制严格照抄 ManagerBasedRlEnv.step**：每个物理子步 `set_target → write_data_to_sim → sim.step → update(dt)`，
共 `decimation=4` 次。

### 6.2 lane 映射

`lane = world * 22 + duck`。rsl_rl 完全不知道"队伍"存在：它看到 `E*22 = 11264` 个独立 MDP，
每个 14 维动作、1 个标量奖励。**参数共享 = 一个策略跑在很多 lane 上**，PPO 侧是标准单智能体代码。

`SoccerVecEnv` 是**唯一**做 lane ↔ world 重排的地方：
- `dones`：鸭子不能单独 done，整个世界一起重置 → 世界的 done 广播 22 份
- `episode_length_buf`：getter 重复 22 份；setter 只写回 `world*22` 那条（rsl_rl 会 `randint_like` 它）
- `extras["time_outs"]`：必须逐 lane（PPO 的 bootstrap 掩码用它）
- `unwrapped` 转发（runner 的 `save()` 要 `common_step_counter`）

### 6.3 PPO 超参（`rl_cfg.py`）

| 项 | 值 | 理由 |
|---|---|---|
| `gamma` | **0.995**（走路用 0.99） | 50 Hz 下 0.99 有效视界 ~2 s，太短；0.995 ≈ 4 s，正好覆盖"踢→滚→到位" |
| `num_steps_per_env` | 24 | 11264 lane × 24 = 27 万 transition/迭代 |
| `learning_rate` | **3e-4 fixed**（走路用 1e-3 adaptive） | 1e-3 是为"1 鸭/世界 + 稠密奖励"调的 |
| `entropy_coef` | 0.01 | 22 个 agent 共用一个策略，lane 噪声是唯一的行为多样性来源 |
| symmetry | **关闭** | 97D 观测已按队规范化，镜像增强冗余；且角色 one-hot 不对称 |

### 6.4 实跑结果

```
8 世界 (176 lane) × 4 迭代：470–515 steps/s，8.4–9.0 s/迭代
512 世界 (11264 lane)：409 ms/控制步 → 9.8 s/迭代，显存 19.3/23.5 GiB
```

| 项 | 值 |
|---|---|
| 构建（22 鸭 × 512 世界） | 7.8 s |
| `set_joint_position_target` ×22 | 0.21 ms |
| `write_data_to_sim` ×22（BAM 逐环境写摩擦，每子步） | **11.6 ms** |
| `sim.step`（512 世界） | 82 ms |
| `sim.forward` | **30.8 ms** → **只在 reset 后调用**（mjlab 自己在文档里说一个子步的滞后是一致的、无害的） |
| 观测 + 奖励（纯 torch） | 3.8 + 1.5 ms |

### ⚠️ 本轮修掉的两个真 bug（都会静默毁训练）

1. **张量别名把 `done` 清零**：`truncated = timed_out` 与 `term_reason["timeout"]` 指向同一张量，
   `_reset_idx` 里 `v[env_ids] = False` 原地把它清零 → `done` 恒为 False、`extras["time_outs"]` 恒 False
   → **PPO 的超时 bootstrap 掩码全程失效**（价值函数学错终止值，且不报错）。修法：`done` 在 reset 之前算，
   且 `term_reason` 存 clone。
2. **奖励量级失衡把策略推进"什么都不做"的坑**：初版 `FALLEN=2.0`、`SPACING=1.0`，是任务梯度的 **300 倍**。
   随机策略 t=0 就倒地、1500 步全程躺着 → 回报恒为 **−3000**，value loss 从 0.002 涨到 10 → 131。
   按"每步值多少"重排后（见 `rewards.py` 的 REWARD MASS 注释）：
   走路推进球 **+0.006/步**、一脚干净踢球 **+3.5/步**、进球 **+50**、倒地 **−0.02/步**、扎堆 **−0.0125/步**。
   重跑：value loss **0.001–0.004**，稳定。

---

## 7. P5 — 训练课程（S0 ✅ 跑通，S1–S5 待跑）

| 阶段 | 队形 | worlds | 迭代 | 目标 | 状态 |
|---|---|---|---|---|---|
| **S0** | 1v1 | 2048 (4096 lane) | 800 | 走过去碰球，球位移 > 0.3 m/回合 | ✅ **达标并进球** |
| S1 | 2v2 | 2048 | 2000 | 双方都能进球，乌龙 < 30 % | 待跑 |
| S2 | 4v4 | 1024 | 2000 | 进球率不塌，出现传球/抢断 | 待跑 |
| S3 | 6v6 | 1024 | 1500 | 无大面积倒地，分区生效 | 待跑 |
| S4 | 11v11 | 512 | 2000–3000（**9.8 s/迭代 → 5.4–8.2 h**） | §8 指标 | 待跑 |
| S5 | 11v11 + 全量 DR + 观测噪声 | 512 | 1000 | 噪声下不退化 | 待跑 |

### 7.1 S0 实测结果（首次训出进球）

`scripts/train_soccer.py --n-team 1 --num-worlds 2048 --iterations 800`（≈14 min，1.0 s/迭代）

| 指标 | 300 iter | **798 iter** |
|---|---|---|
| 有触球的回合占比 | 98.5 % | **94.7 %** |
| 每回合触球次数 | 4.32 | **3.56** |
| **每回合进球** | 0 | **0.147（75 回合进 11 球）** |
| 球每回合位移 | 4.35 m | **4.00 m** |
| 首次触球（中位） | 1.08 s | **0.90 s** |
| 鸭-球最近距离 | 0.077 m | 0.082 m |
| 倒地时长占比（>60°） | 0.40 | **0.36** |
| 回合结束原因 | stalemate 53 % / timeout 47 % | goal 15 % / stalemate 61 % / timeout 24 % |
| 奖励 | 每回合 +0.06 | **每回合 +1.53** |

**S0 验收线（球位移 > 0.3 m/回合）远超**：4.0 m。

### 7.2 S0 暴露并修掉的问题（按影响排序）

1. **`action_scale=0.25` rad 太小 —— 主因。** 仓库所有配方都是 `scale=1.0`（±1 动作 = ±1 rad = 57° 关节偏移）。
   0.25 rad 让鸭子**物理上迈不出一步**，于是它学会了"身体前倾 51° 去够球"
   （实测平均投影重力 z = **−0.63**，即倾斜 ~51°），触球率 10 %、倒地 78 %、球位移 0.27 m。
   改成 1.0 后 300 迭代：触球率 **10 % → 98.5 %**，球位移 **0.27 → 4.35 m**，倒地 **0.78 → 0.40**。
2. **1v1 出生点误用 11 人阵型。** `events.spawn_poses()` 没有 `n_team` 参数，内部读 `S.formation()`（默认 11 人）
   → 1v1 的两只鸭被摆到 **11 人制的门将线**（离球 2.83 m），朝向却取自 1v1 阵型。
   每回合 5 s 僵局结束，鸭子连球都够不到。修好后最近距离 2.83 → **0.64 m**。
3. **奖励缺"走向球"这一项（设计缺口）。** 原来唯一的稠密任务项是"球→对方球门"，而它**必须先碰到球**才可能发生——鸡生蛋。
   300 迭代实测：0 触球、0 进球、value loss **0.0001**、action std 一路上涨（只剩熵梯度）。
   新增 `ball_approach`：球队**最近那只鸭**到球的距离势差（不可 farm：不动 = 恰好 0），权重与"球→门"同量级（15）。
4. **1v1 出生距离 0.8 m → 0.45 m**：球进入"一次迈步 + 一次踉跄"的射程。

### 7.3 熵系数 / init_std 的 A/B（已解决 std 饱和）

问题：`entropy_coef=0.01` + `init_std=1.0` 时，动作 std 从 1.0 一路涨到 **4.19**（iter 800），
而动作被 clamp 到 ±1 → 约 80 % 的采样落在 clamp 上，**PPO 训练的 log-prob 与环境实际执行的动作不一致**。

A/B（S0，300 迭代，其余完全相同）：

| entropy | init_std | std@300 | 触球率 | 球位移/回合 | 倒地时长 | 进球/回合 | value loss |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.0 | 1.92 | 98.5 % | 4.35 m | **40.0 %** | 0 | 0.0079 |
| **0.003** | **1.0** | **1.03** | **100 %** | 3.69 m | **19.3 %** | **0.046** | 1.35 |
| 0.01 | 0.5 | 1.18 | 100 % | 1.64 m | **14.2 %** | 0 | 0.0018 |
| 0.003 | 0.5 | 0.46 | 100 % | 4.61 m | 19.4 % | 0 | 0.0038 |

**结论：默认改成 `entropy_coef=0.003`，`init_std` 保持 1.0。**
理由：std 稳在 ~1（动作仍是全量程，探索真实存在），倒地时间减半，而且**300 迭代就出现进球**
（基线要 798 迭代）。`init_std=0.5` 把探索压得太窄，与 0.003 耦合后 std 只剩 0.46 ——
对后面要打 2v2 / 11v11 的课程来说起点太保守。

### 7.4 S0 尚未解决的问题（下一轮）

| 现象 | 数据 | 诊断 / 下一步 |
|---|---|---|
| 倒地时长 19–36 % | 平均倾斜 ~51°，60° 才罚 | 熵系数修正后已减半；若仍偏高，先看录像区分"爬行"与"摔"，**不要**直接加重惩罚（会回到"什么都别做"） |
| 61 % 回合僵局结束 | 5 s 无进展 | 球可能卡在墙角/门柱；僵局判定可加入"球是否贴边" |
| value loss 1.35（ent 0.003） | 进球 ±50 使回报方差大 | 可接受；若涨到 10+ 查奖励量级 |

### 7.5 S0 长跑（1200 迭代）结果与诊断 —— 发现"戳球农场"

`--n-team 1 --num-worlds 2048 --iterations 1200`（≈28 min）。每 300 迭代评测 96 回合：

| iter | 进球/回合 | 球位移/回合 | 触球/回合 | 倒地时长 | 僵局结束 | 超时结束 |
|---|---|---|---|---|---|---|
| 300 | 0.020 | 4.26 m | 6.49 | 0.184 | 38.8 % | 59.2 % |
| 600 | 0.042 | 4.79 m | 6.43 | 0.368 | 35.4 % | 60.4 % |
| **900** | **0.051** | 5.22 m | 7.40 | 0.294 | 24.5 % | 70.4 % |
| 1199 | **0.010** | **7.28 m** | 8.33 | 0.393 | 18.6 % | 80.4 % |

**结论：S0 的动作技能过线，但 900 迭代后开始"跑偏"。**

- 过线的部分：鸭子 100 % 的回合能碰到球、每回合 6–8 次触球、球位移 4.3 → **7.3 m**、僵局从 39 % 降到 19 %。
  S0 的验收线（球位移 > 0.3 m/回合）早就超了。
- **跑偏的部分：进球从 900 的 0.051 掉到 1199 的 0.010（5 倍），而触球/球位移还在涨。**
  这是典型的**"戳球农场"**：`touch`（每次 0.15）+ `ball_approach` 会为"把球乱捅一气"付钱，
  而进球需要把球朝 +x 推 3.5 m 进 0.7 m 的门 —— 难得多。
  实测鸭子平均倾斜 **53°**（不是走路，是趴着/爬着捅球），均值动作幅度 0.93（接近 clamp）。
- `action std` 从 1.0 漂到 **2.10**（熵 0.003 后比原来的 4.19 好，但仍在涨）。

**下一轮的修法（按优先级）**：

1. `touch` 只在**把球朝进攻方向推动**时计入（`Δball_x · attack_sign > 0`），或把权重从 0.5 降到 0.2。
2. `ball_goal_progress` 权重从 15 提到 25–30，让"往门里推"压过"乱捅"。
3. 倒地惩罚暂时**不动**（36–39 % 的倒地正是"爬着捅球"的副产品；先改上面两条再看它降不降）。

### 7.6 长跑里的 NaN 崩溃（已修）

第 717 迭代整次训练崩溃：`ValueError: The rewards returned by the environment contain NaN values`。

根因有两处，都修了：

1. `ball_approach_progress` 里 `inf - finite = inf`，再乘 one-hot 的 0 得到 **`inf × 0 = NaN`**。
2. 更普遍地：接触求解偶发一步发散时，位置是 NaN，`cdist`/`clamp` 一路传下去。

对策（照仓库惯例，"每个 reward 函数自己 nan_to_num 安全"）：
- `ball_approach_progress` 在 one-hot 相乘**之前** `nan_to_num`；
- 所有 term（spacing / fallen / goal_progress / touch / no_progress）返回值都过 `nan_to_num`；
- env 层再加一道 per-term 守卫并**计数**（`extras["nan_reward_terms"]`），只清洗不静默。
- 新增测试：`test_all_terms_are_finite_on_a_diverged_state`（全 NaN 状态，每个 term 必须有限）、
  `test_approach_reward_survives_inf_and_nan_inputs`。

实测：修好后 1200 迭代跑满、`nan_reward_terms` 全程为空（每步都检查）。

### 7.7 小规模阶段的运行配方

```bash
# S0 1v1（已验证）
python scripts/train_soccer.py --n-team 1 --num-worlds 2048 --iterations 800 \
    --wandb offline --experiment-name soccer_S0

# 评测任何 checkpoint
python scripts/soccer_eval.py --checkpoint logs/soccer/soccer_S0/<ts>/model_798.pt \
    --n-team 1 --episodes 64
```

worlds 的选择按 `nconmax × worlds ≈ 500k` 维持显存：1v1（nconmax 300）可跑 2048，11v11（1100）只能 512。

---

## 8. P6 — 评测（🟡 `scripts/soccer_eval.py` 已可用，CSV / 对手池未做）

| 指标 | 门槛 |
|---|---|
| 球在场内时间占比 | > 60 % |
| 进球率 | 0.7–3 / 90 s |
| 乌龙占比 | < 10 % |
| 单局倒地总时长 | < 20 % 场时 |
| 僵持（球 5 s 位移 < 0.15 m 且 ≥3 只在 0.25 m 内） | < 2 次/局 |
| 平均每鸭触球 | > 2 次 |

- [x] **P6-T0** `scripts/soccer_eval.py`：加载 checkpoint → 并行跑 N 局 → JSON 表
  （进球/球位移/触球/首次触球/倒地时长/回合长度/结束原因/鸭-球最近距离）。S0 就是靠它定位问题的。
- [ ] **P6-T1** CSV 批量输出（多 checkpoint × 多 seed）+ 乌龙/控球秒数统计
- [ ] **P6-T2** 对手池：冻结 checkpoint 对战（league-lite），检测自博弈退化

---

## 9. P7 — 展示

- [ ] **P7-T1** `scripts/soccer_view.py`：viewer 回放 + HUD（比分/球速/角色）
- [ ] **P7-T2** 录像（`--video`）

---

## 10. 常量总表

`src/mjlab_microduck/soccer/pitch.py`（几何与物理）：

```python
PITCH_L, PITCH_W        = 7.0, 4.5
CENTER_CIRCLE_R         = 0.61
PENALTY_D, PENALTY_W    = 1.10, 2.67
GOAL_W, GOAL_H, GOAL_D  = 0.70, 0.35, 0.25
POST_HALF               = 0.010          # 20 mm 方柱
WALL_H, WALL_MARGIN     = 0.20, 0.30
BALL_RADIUS             = 0.035
BALL_ROLL_LOSS          = 2.15e-4        # freejoint frictionloss, 6 dof
SIM_TIMESTEP            = 0.005
CONTROL_DECIMATION      = 4              # 50 Hz
EPISODE_S               = 30.0           # S0-S3
EPISODE_S_11V11         = 40.0
NJMAX, NCONMAX          = 2600, 1100     # 22 鸭实测 nefc 2017
NUM_DUCKS_PER_TEAM      = 11
NUM_WORLDS              = 512
```

`src/mjlab_microduck/soccer/env.py::SoccerEnvCfg`（环境）：

```python
n_team              = 11      # 1=S0, 2=S1, 4=S2, 6=S3, 11=S4
action_scale        = 1.0     # rad/unit —— 仓库惯例；0.25 会让鸭子迈不出步
clip_actions        = 1.0
touch_radius        = 0.16    # 鸭-球中心距，代替接触传感器
critic_extra_dim    = 10
njmax / nconmax     = None    # None -> 按 n_team 缩放
```

`src/mjlab_microduck/soccer/rewards.py`（权重；惩罚为正权重，见 `REWARD MASS` 注释）：

```python
GOAL_WEIGHT = 50.0            BALL_PROGRESS_WEIGHT = 15.0
BALL_APPROACH_WEIGHT = 15.0   TOUCH_WEIGHT = 0.5
SPACING_WEIGHT = 0.05         FALLEN_WEIGHT = 0.02
NO_PROGRESS_WEIGHT = 0.5
TOUCH_BONUS = 0.3   TOUCH_COOLDOWN_S = 5.0
SPACING_MIN_M = 0.25   FALLEN_TILT_DEG = 60.0
NO_PROGRESS_WINDOW_S = 5.0   NO_PROGRESS_MIN_MOVE = 0.15
```

`src/mjlab_microduck/soccer/rl_cfg.py`（PPO）：`gamma=0.995  lam=0.95  lr=3e-4(fixed)
num_steps_per_env=24  entropy_coef=0.003  init_std=1.0  minibatch=4  epochs=5`

---

## 11. 风险与止损

| 风险 | 触发信号 | 止损动作 |
|---|---|---|
| **戳球农场**（已发生） | 触球/球位移在涨，**进球数却不涨或下跌**（S0：0.051@900 → 0.010@1199） | `touch` 只在球被朝进攻方向推动时计入；调高 `ball_goal_progress` 权重 |
| **奖励量级失衡**（已发生） | value loss 从 0.00x 涨到 10–100+ | 按"每步值多少"重排权重，别按直觉设 1.0/2.0 |
| 对称僵局（都站着不动） | S1 后触球/回合数不涨，总分靠 0 撑着 | 开场不对称（球±0.3 m + 逐鸭抖动 + 随机球权）+ 无进展惩罚 |
| 22 只挤成一团 | 队友间距均值 < 0.3 m，倒地时长暴涨 | 反扎堆权重 ×3；检查角色 one-hot 是否被策略忽略 |
| 训练太慢 | 单迭代 > 12 s | 降 worlds（`nconmax × worlds ≈ 500k`）；**不要**指望降 `njmax` 省显存 |
| 仿真偶发发散 | `extras["nan_reward_terms"]` 非空 | 保持 reward 全 NaN-safe；查 `nan_state` 终止是否触发 |
| conda 环境漂移 | `uv run` 与 conda 版本不一致 | 只改 `pyproject.toml` → `uv lock` → 重跑 `sync_conda.sh` |

---

## 12. 踩坑清单（按代价排序，全部本仓库真实发生）

> 这张表是给下一个接手的 agent 看的。前 6 条都**不会报错**，只会让训练"怪怪的"。

### 会让整次训练崩溃

| # | 坑 | 症状 | 对策 |
|---|---|---|---|
| 1 | freejoint 的 `qpos0` 来自 spec 里 root body 的 pose；22 只鸭都留在原点 | `mj_stackAlloc: stack overflow`（8 万接触），**迭代 0 之前** | `duck_spec(i)` 里把 `trunk_base` 摆到出生位姿 |
| 2 | 想用一个 spec 装 22 只鸭 | mjlab 把另外 21 个 freejoint 当铰链，静默毁模型 | 22 个独立 `EntityCfg` |
| 3 | `truncated = timed_out` 与 `term_reason[...]` 是同一张量，reset 时原地清零 | `done` 恒 False、`extras["time_outs"]` 恒 False → **PPO 超时 bootstrap 全程失效** | `done` 在 reset 之前算；`term_reason` 存 clone |
| 4 | `inf × 0 = NaN`（势差里）+ 发散状态未清洗 | 训练跑到第 717 迭代 `rewards contain NaN` 整次崩溃 | 所有 reward 函数自带 `nan_to_num`；env 层再守一道并计数 |

### 会让训练静默跑偏

| # | 坑 | 症状 | 对策 |
|---|---|---|---|
| 5 | 标定球时初始自转方向写反（`ω_y = -v/r`） | 球在**打滑**不是滚动，1 s 掉到 0.19 m/s 再拖长尾，伪装成"滚动阻力" | 沿 +x 滚动要 `ω_y = +v/r`；标定脚本必须断言 |
| 6 | 手拼的球丢了 `<inertial>` → MuJoCo 按密度算成 **180 g**（真球 15 g） | 任何 `frictionloss` 数值都失去意义 | 必须 `MjSpec.from_file(ball.xml)`，测试断言 `body_mass == 0.015` |
| 7 | `einsum("ij,enj->eni")` 把"球员"和"坐标"两个轴搞混（`pos_xy` 是 `(E,N,2)`，中间轴才是球员） | 队形质心全错 | 写观测前先写对称性测试 |
| 8 | 180° 合成四元数分量序取错（(w,x,y,z) 里取了第 3/2/1/4 个） | 变成**绕 y 转 180°**；用单位四元数验证完全看不出来（退化案例） | 用非退化 yaw 验证 `R(qz⊗q) == Rz(π)R(q)` |
| 9 | 净胜球/上次触球是世界标量直接广播给 22 条 lane | 两队读同一符号 → 破坏 180° 对称性，"我领先吗"变成噪声 | 队相对量乘 `team_sign` |
| 10 | `fallen_penalty` 的 clamp 方向反了（`onset - gz` 应为 `gz - onset`） | 站直付满额、躺下付 0（正好反了） | 用 0/30/59/61/90° 的参数化测试 |
| 11 | 奖励量级失衡：惩罚是任务梯度的 **300 倍** | value loss 0.002 → 10 → **131**，"什么都不做"成了最优策略 | 按"每步值多少"排 `REWARD MASS`（见 `rewards.py` 头注释） |
| 12 | 只有"球→球门"的稠密项，而它必须先碰到球 | 0 触球、0 进球、value loss **0.0001**、只剩熵梯度 | 补 `ball_approach`（最近那只鸭到球的距离势差） |
| 13 | `action_scale=0.25` rad（仓库惯例 1.0） | 鸭子**物理上迈不出步**，学会前倾 51° 趴着够球；触球率 10 % | 用仓库惯例 1.0；改后 300 迭代触球率 → 98.5 % |
| 14 | 熵系数 0.01 + 动作 clamp ±1 | 动作 std 漂到 **4.19**，~80 % 采样撞 clamp，log-prob 与实际动作不一致 | `entropy_coef=0.003`（std 稳定在 ~1，倒地减半） |
| 15 | 硬编码 `N=22` 漏进 `extras` / `spawn_poses` 漏传 `n_team` | 任何非 11v11 的队伍规模都错（1v1 甚至被摆到门将线，离球 2.83 m） | 队伍规模必须全链路参数化，并用"1v1 出生距离"测试锁死 |

### 环境 / 性能

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| 16 | `nconmax` 才是显存杀手，不是 `njmax` | 512 世界 `nconmax=1600` 直接 CUDA graph OOM；`njmax` 2200→2400 几乎不占 | 512 世界用 `njmax=2600 / nconmax=1100`；worlds 按 `nconmax × worlds ≈ 500k` 缩放 |
| 17 | `sim.forward()` 每控制步 30.8 ms（占 8 %） | 512 世界每步多花 30 ms | 只在 reset 之后调用（mjlab 自己文档说一子步滞后是一致的、无害的） |
| 18 | `uv export` 首行 `-e .` 让 `uv pip` 碰到 bam 的 URL 冲突（branch vs commit） | `Requirements contain conflicting URLs` | 导出时滤掉 `-e .`，装完依赖再 `--no-deps -e .` |

## 13. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-10-29 | 初版；P0 环境重建（py3.12），场地 7.0×4.5，allcollisions，`njmax=1850`，球改用 freejoint `frictionloss`（撤回 `condim=6` 方案） |
| 2026-10-29 | **进度总览 + 踩坑清单落档**：新增「进度总览」（阶段完成度 / 代码地图 / S0 数字 / 未解决问题 / 下一步）与「踩坑清单」（18 条，按代价排序：4 条崩溃、11 条静默跑偏、3 条环境性能） |
| 2026-10-29 | **S0 长跑 1200 迭代**：动作技能过线（触球 100 % / 6–8 次/回合 / 球位移 7.3 m），但 900 迭代后出现**戳球农场**（进球 0.051 → 0.010，触球却还在涨；鸭子平均倾斜 53° 趴着捅球）。修掉一个会让整次训练崩溃的 NaN bug（`inf×0` + 发散状态未清洗，新增 3 处守卫与 2 条测试）。熵系数 A/B 后定为 0.003（std 4.19 → 2.10，倒地减半） |
| 2026-10-29 | **S0 跑通并首次进球**：800 迭代（14 min）→ 触球率 94.7 %、每回合 3.56 次触球、**0.147 球/回合**、球位移 4.00 m。修 3 个问题：`action_scale` 0.25→**1.0**（主因：迈不出步，改前触球率 10 %）、`spawn_poses` 漏传 `n_team`（1v1 被摆在门将线）、奖励缺 `ball_approach`（无"走向球"梯度 → value loss 0.0001）。新增 `scripts/soccer_eval.py`（P6-lite） |
| 2026-10-29 | **P4 完成**：`soccer/env.py`（自持 step 循环的 `SoccerEnv`）+ `vecenv.py`（`E*22` lane）+ `rl_cfg.py` + `scripts/train_soccer.py`；PPO 实跑通（8 世界/512 世界）。修掉张量别名导致 `done` 恒 False（PPO 超时 bootstrap 失效）与奖励量级失衡（value loss 0.002→131）两个静默 bug。实测 409 ms/控制步、9.8 s/迭代 |\n| 2026-10-29 | **P3 完成**：`soccer/obs.py`（97D，体坐标系 + 队相对世界量）、`rewards.py`（势差型不可farm进度项）、`events.py`；47 条新测试。修掉 4 个静默 bug（einsum 轴序、四元数分量序、世界标量直接广播、clamp 方向反）。全套 367 条绿 |
| 2026-10-29 | **P2 完成（世界部分）**：`soccer/scene.py`，22 个独立 Entity，15 条测试全绿。新实测：nefc **2017**（→`NJMAX=2600`）、`nconmax=1100`、512 世界 **331 ms/控制步**（迭代 7.9 s，远高于原估 2.07 s）、本体显存 12.6 GB；记录两个坑（`qpos0` 必须=阵型否则 stack overflow；`nconmax` 才是显存杀手） |
| 2026-10-29 | **P1 完成**：`soccer/pitch.py` + 30 条测试全绿。修掉两个标定错误（初始自转方向错 → 打滑；标定球丢了 `<inertial>` → 180 g 而非 15 g），常量 `4e-4 → 2.15e-4`，停止时间 19 s → **3.76 s**；网箱改为实心（`contype=0` 无法只挡鸭不挡球） |
| 2026-10-29 | **P0 完成**：conda py3.12.16 + torch 2.9.1+cu128 + warp 1.12.0，5 门全绿；新增 `docs/soccer_env.md` 快照；`sync_conda.sh` 解决 bam URL 冲突（滤掉 `-e .` + `--no-deps -e .`），幂等重跑 1.4 s |
