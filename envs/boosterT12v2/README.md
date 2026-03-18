# boosterT12v2

`boosterT12v2` 是一个独立于旧单机器人任务的 IsaacGym 2v2 Booster T1 环境骨架。

当前目标不是复刻 `passBall / trapBall / chaseBall` 的任务语义，而是先提供一个：

- 位于 `envs/` 下的正式环境目录
- 使用 `resources/T1` 原始资产与控制参数
- 通过独立的多人底层控制器创建 `4` 台 T1 + `1` 个球 + 标准场地
- 可被 `scripts/runner.py` 以新入口拉起
- 能输出多人 locomotion obs 和 team-level infer state

## 文件

- [BoosterT12v2Env.py](/home/ubuntu/jrWork/booster_gym/envs/boosterT12v2/BoosterT12v2Env.py)
  新环境壳。负责多人 buffer、team 观测、高层命令入口、step/reset。
- [BoosterT12v2Env.yaml](/home/ubuntu/jrWork/booster_gym/envs/boosterT12v2/BoosterT12v2Env.yaml)
  环境配置。保留 T1 资产与控制参数，同时加入 `game` 段定义 2v2 布阵。
- [MultiAgentLowLevelController.py](/home/ubuntu/jrWork/booster_gym/envs/components/MultiAgentLowLevelController.py)
  新的多人底层控制器。与旧 `LowLevelController` 并列，不修改旧控制器行为。

## 当前已实现

- 4 台 T1 的场景创建与固定出生位
- 单球与场地创建
- 多机器人 DOF/刚体张量组织
- batched locomotion observation 计算
- 面向 infer 的 team-level state 导出
- `Runner.boosterT12v2()` 的最小启动入口
- `Runner.boosterT12v2Locomotion()` 的低层运动 smoke test
- 默认会通过 `basic.checkpoint: -1` 自动加载 `logs/low/**/*.pth` 下最新的低层 locomotion 权重

## 当前未完成

- 比赛规则与 reward
- 球权、碰撞语义、射门/出界/重置逻辑
- 真正的 2v2 高层动作接口设计
- 训练链路适配
- IsaacGym 运行态 smoke test 结果确认

## 设计约束

- 不修改旧 `LowLevelController` 的单机器人假设
- 不把 2v2 特性塞进旧 `passBall / trapBall / chaseBall`
- 公共物理参数继续复用 `resources/T1` 和现有 yaml 风格
