# 足球训练环境快照（conda `microduck_rl`）

> 任务清单 / 进度 / 训练结果：见 `docs/soccer_tasks.md`。本文件只讲环境怎么装、怎么验证。

> P0 的验收记录。**重建方式只有一条**：`bash scripts/sync_conda.sh`（`--recreate` 会先删库）。
> 任何"pip 装一下缺的包"的补救都会偏离 `uv.lock`，下次重建即失效。

---

## 1. 已验证的快照（2026-10-29，P0 验收通过）

| 项 | 值 |
|---|---|
| conda env | `microduck_rl` @ `/home/kaisong.shi/miniconda3/envs/microduck_rl` |
| Python | **3.12.16** |
| torch | **2.9.1+cu128**（`torch.version.cuda == 12.8`，`cuda_available=True`，`device_count=1`） |
| warp-lang | **1.12.0**（CUDA Toolkit 12.9 / Driver 13.0，`wp.get_device('cuda') → cuda:0`） |
| mujoco | 3.10.0 |
| mujoco-warp | 3.8.1 |
| mjlab | 1.3.0 |
| rsl-rl-lib | 5.0.1 |
| better-actuator-models (bam) | 1.0.1，git `62bd8ce1`（branch `mjlab_frictionloss` 的锁定提交） |
| numpy | 2.4.1 |
| onnx | 1.23.2 |
| protobuf | 6.33.6 |
| GPU | NVIDIA GeForce RTX 4090, 24 GiB, driver **580.167.08** |
| 仓库 commit | `bae39f5` |

包总数 154（`uv export --no-hashes --no-dev`）。

---

## 2. 验收门（全部通过）

| 门 | 命令 | 结果 |
|---|---|---|
| **G1** | `python -c "import sys,torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"` | `3.12.16 2.9.1+cu128 12.8 True 1` ✅ |
| **G2** | `python -c "import warp as wp; wp.init(); print(wp.get_device('cuda'))"` | `cuda:0`（RTX 4090, sm_89, mempool） ✅ |
| **G3** | `which train; train 2>&1 \| head -5` | `.../envs/microduck_rl/bin/train`，任务表列出 `Mjlab-BallKick-Flat-MicroDuck` 等 ✅ |
| **G4** | `train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --gpu-ids '[0]'` | 5 迭代跑完，0.51 s/迭代，`Episode_Reward/*` 正常（`head_pose_tracking: 0.0621`），`nan_state: 0` ✅ |
| **G5** | `python -c "import bam.model, mujoco, mujoco_warp"` | `ok 3.10.0` ✅ |

**幂等性**：在没有变更时重跑 `scripts/sync_conda.sh` 用时 **1.4 s**，不安装任何包（`Resolved 154 packages`，全部已满足）。

---

## 3. `scripts/sync_conda.sh` 处理的四个坑

1. **`pip install -e .` 不还原依赖真源。** bam 的元数据钉 `protobuf<4.0` → onnx 级联降到 1.17.0（无 wheel，源码编译失败），并且拉入坏桩包 `zmq==0.0.0`。三者都由 `pyproject.toml` 的 `[tool.uv] override-dependencies` 中和 —— **只有 `uv` 读这个字段**，所以必须走 `uv export` + `uv pip install`。
2. **`-e .` 的 bam URL 冲突。** 导出的 requirements 首行是 `-e .`；`uv pip` 会把它当项目解析，而项目元数据按 **branch** 写 bam（`mjlab_frictionloss`）、锁文件按 **commit** 钉（`62bd8ce1`）。两者指同一物，但 `uv pip` 直接报
   `Requirements contain conflicting URLs for package better-actuator-models`。
   **对策**：导出时 `grep -v -x -e '-e .'` 去掉，装完依赖后再
   `uv pip install --no-deps -e .`（依赖已全部就位，只落 console script 与 `mjlab.tasks` entry point）。
3. **不装 conda 的 cudatoolkit / cudnn / pytorch。** warp 1.12 自带 CUDA 12.9、torch 的 cu12 wheel 自带运行时，混装是 `libcudnn not found` / CUDA 版本错配的常见来源。
4. **`GLIBCXX_3.4.30 not found`。** 若出现（旧 `libstdc++`），`conda install -c conda-forge "libstdcxx-ng>=13"`，或先 `unset LD_LIBRARY_PATH`。当前 conda `libstdcxx 15.2.0`，无此问题。

---

## 4. 日常用法

```bash
conda activate microduck_rl
cd /home/kaisong.shi/code/microduck_rl

# 1) 改了 pyproject.toml 之后：先锁，再同步
uv lock
bash scripts/sync_conda.sh

# 2) 环境损坏 / 换机器
bash scripts/sync_conda.sh --recreate

# 3) 检查 CUDA 是否真的活着
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 4) 足球任务（不经过 mjlab 的 task registry，见 soccer/env.py 开头的说明）
python scripts/train_soccer.py --n-team 1 --num-worlds 2048 --iterations 800 \
    --wandb offline --experiment-name soccer_S0
python scripts/soccer_eval.py --checkpoint logs/soccer/soccer_S0/<ts>/model_798.pt \
    --n-team 1 --episodes 64
```

> **注意**：仓库里仍有 `.venv`（`uv run` 用的）。**训练足球任务请用 conda env**，不要混用：
> 两者的包版本相同（都以 `uv.lock` 为真源），但 `uv run` 会重新解析并可能触发网络访问；
> conda env 是离线可用的那一份。
