# BoosterGym

BoosterGym 是一个面向机器人足球控制的研究型代码仓库，核心思路是把策略分成三层：

- Low-level：面向真实机器人执行的底层运动控制
- Mid-level：基于物理仿真的技能策略
- High-level：基于抽象博弈环境的多人战术策略

这三层分别在不同环境中训练和验证，但通过统一接口衔接起来。

## 当前进展

目前仓库里已经形成的主线包括：

- Isaac Gym 中的中层技能训练
- `hyperGym` 这套轻量 2D 足球战术环境
- 基于 VLM 的 `hyperGym` 轨迹采集
- 基于 VLM 数据的行为克隆
- 从 BC 初始化出发的 high-level 对抗强化学习
- `core/checkpoints/` 统一管理当前工作流需要复用的权重

这个仓库仍然是研究工作区，不是整理完毕的开源 release。

## 三层架构

### 1. Low-Level

底层负责真实机器人可执行的稳定运动控制。

项目中的定位是：

- 面向真机部署
- 提供稳定、鲁棒、可复用的运动能力
- 上层不直接操纵关节，而是给子目标或命令

### 2. Mid-Level

中层负责在物理仿真中学习技能策略，把“技能类型 + target”转成可执行行为。

当前重点技能包括：

- `chaseBall`
- `passBall`
- `trapBall`

相关位置：

- [envs/chaseBall](/home/ubuntu/jrWork/booster_gym/envs/chaseBall)
- [envs/passBall](/home/ubuntu/jrWork/booster_gym/envs/passBall)
- [envs/trapBall](/home/ubuntu/jrWork/booster_gym/envs/trapBall)
- [scripts/train.py](/home/ubuntu/jrWork/booster_gym/scripts/train.py)
- [scripts/runner.py](/home/ubuntu/jrWork/booster_gym/scripts/runner.py)
- [core/checkpoints/mid_level](/home/ubuntu/jrWork/booster_gym/core/checkpoints/mid_level)

### 3. High-Level

高层负责在简化的多人足球状态上做战术决策，输出“当前应该执行什么技能、打到哪里”。

这一层目前主要基于 `hyperGym`：

- 一个轻量级 2D 足球博弈原型环境
- 不模拟复杂关节接触
- 更关注多人协同、传球、停球、争球和进攻拓扑

当前高层动作接口统一为：

```python
{
  "skill": "move" | "pass" | "trap",
  "target": np.ndarray(shape=(2,))
}
```

在 2v2 设定下，策略输入四个球员和球的全局状态，但只输出一个队伍两个人的动作。

当前高层观测是一个 team-conditioned 的 `29` 维状态向量，包含：

- 4 个球员的状态
  - `x, y`
  - `vx, vy`
  - `is_controlled_team`
  - `has_ball`
- 球的状态
  - `ball_x, ball_y`
  - `ball_vx, ball_vy`
  - `owner_flag`

当前高层动作可以理解成：

- 每个队伍输出两个 `(policy_id, target_xy)`
- `policy_id ∈ {move, pass, trap}`
- `trap` 也保留 `target_xy`，但它的语义是“截球/接球位置”，不是传球目标

相关位置：

- [envs/hyperGym](/home/ubuntu/jrWork/booster_gym/envs/hyperGym)
- [scripts/hlp](/home/ubuntu/jrWork/booster_gym/scripts/hlp)
- [docs/hypergym_vlm_bc_rl](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl)

## 当前 High-Level Pipeline

当前高层训练主线是：

1. 在 `hyperGym` 中调用 VLM 进行战术决策并采集轨迹。
2. 过滤出可用的 VLM 样本，训练 state-only 的 BC policy。
3. 用 BC policy 初始化 high-level RL。
4. 先以冻结的 BC policy 作为初始对手。
5. 只有最近 win rate 达到 `0.8` 后，才解锁 FSP 风格的历史对手池。
6. 用 dense reward shaping 和 staged self-play 继续优化传球和进球能力。

对应文档：

- [dataset.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/dataset.md)
- [bc.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/bc.md)
- [hlp_selfplay.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/hlp_selfplay.md)
- [experiment_log.md](/home/ubuntu/jrWork/booster_gym/docs/hypergym_vlm_bc_rl/experiment_log.md)

## 仓库结构

主要目录：

- [envs](/home/ubuntu/jrWork/booster_gym/envs)：环境定义，包括 Isaac Gym 任务和 `hyperGym`
- [core/agents](/home/ubuntu/jrWork/booster_gym/core/agents)：可复用 RL agent
- [core/imitation](/home/ubuntu/jrWork/booster_gym/core/imitation)：行为克隆相关数据与模型
- [core/checkpoints](/home/ubuntu/jrWork/booster_gym/core/checkpoints)：当前工作流复用的 low/mid/high level 权重
- [scripts](/home/ubuntu/jrWork/booster_gym/scripts)：保留的稳定入口
- [scripts/tmp](/home/ubuntu/jrWork/booster_gym/scripts/tmp)：实验性脚本和临时工作流
- [scripts/hlp](/home/ubuntu/jrWork/booster_gym/scripts/hlp)：high-level self-play RL
- [docs](/home/ubuntu/jrWork/booster_gym/docs)：实验记录和说明文档
- [deploy](/home/ubuntu/jrWork/booster_gym/deploy)：部署相关代码和资源

## 快速开始

### 中层技能训练

示例：

```bash
python scripts/train.py --checkpoint -1 --task=PassBallEnv
```

### High-Level 行为克隆

示例：

```bash
python scripts/tmp/hypergym_vlm_bc_rl/train_bc.py \
  --dataset-roots datasets/vlm_vs_vlm_qwen3vl_2v2_deadball_ep1000 \
  --only-vlm \
  --save-dir core/checkpoints/high_level/bc/run
```

### High-Level 自博弈强化学习

示例：

```bash
python scripts/hlp/train_selfplay.py \
  --updates 400 \
  --save-dir core/checkpoints/high_level/rl/run \
  --bc-init-checkpoint core/checkpoints/high_level/bc/bc_policy_stage1_gpu_resume40/best.pt
```

## 依赖说明

这个仓库当前默认运行在研究环境里，而不是完整封装的安装环境里。

至少包括：

- Python
- PyTorch
- TensorBoard / WandB
- Isaac Gym

[requirements.txt](/home/ubuntu/jrWork/booster_gym/requirements.txt) 目前只覆盖了很小一部分依赖，不能视作完整环境描述。

## 说明

- `datasets/` 已加入 `.gitignore`，默认用于存放大体量采样数据
- `core/checkpoints/` 已加入 `.gitignore`，用于存放当前工作流需要复用的权重资产
- `logs/` 存放 checkpoint 和实验输出
- [scripts/tmp](/home/ubuntu/jrWork/booster_gym/scripts/tmp) 下很多脚本是快速实验产物，变化会比较快

## 参考文档

本 README 的总体提炼主要参考：

- [boosterGym-NIPS2026.docx](/home/ubuntu/jrWork/booster_gym/docs/boosterGym-NIPS2026.docx)

英文版本见 [README.md](/home/ubuntu/jrWork/booster_gym/README.md)。
