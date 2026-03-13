# hyperGym

`hyperGym` 是仓库里一套轻量 2D 足球博弈原型环境，用来快速验证高层策略接口、多人对抗逻辑和可视化回放，不依赖 Isaac Gym viewer。

## 目标

- 给高层策略提供一个足够快、足够可视化的原型环境
- 统一高层动作接口为 `skill + target`
- 支持 `1v1` 和 `2v2` 的小场景对抗
- 用比真实物理更简单、但比纯状态机更保守的方式建模争球、传球、停球

## 当前文件

- [simulation.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/simulation.py)
  核心环境、球员与球的更新、进球判定、碰撞与丢球逻辑
- [policies.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/policies.py)
  当前 demo 用的 scripted policy
- [renderer.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/renderer.py)
  2D 渲染和 MP4 导出
- [main.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/main.py)
  控制器封装和 match/demo 导出入口
- [ball.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/ball.py)
  球的参数和阻尼运动模型
- [player.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/player.py)
  球员的简化移动模型
- [training_interface.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/training_interface.py)
  当前的轻量 reward 逻辑
- [data_interface.py](/home/ubuntu/jrWork/booster_gym/envs/hyperGym/data_interface.py)
  observation 组装

## 动作接口

当前高层动作统一为：

```python
{
    "skill": "move" | "pass" | "trap" | "dribble",
    "target": np.ndarray(shape=(2,))
}
```

当前语义：

- `move`
  只移动人，不带球
- `pass`
  球员靠近球后将球朝 `target` 踢出
- `trap`
  球员靠近球后尝试降低球速，但不会吸附控球
- `dribble`
  当前默认关闭，在环境里退化为 `move`

设计意图：

- 不把“碰到球 = 完美控球”当成默认前提
- 在真实中层 `dribble` 尚不可靠时，不让高层学到过于理想化的控球假设

## 当前抽象

### 球

球是一个 2D 自由球体，不是持球绑定物体。

当前球参数近似参考了 `ballWorld / LowLevelController`：

- `ball_radius = 0.11`
- `ball_density = 80.0`
- `ball_linear_damping = 0.015`
- `ball_angular_damping = 0.01`
- `wall_restitution = 1.0`

自由球更新不是简单的 `velocity *= friction`，而是基于阻尼做微步进更新。

### 球员

球员是简化点质量模型：

- 按 `move_towards(target)` 移动
- 有最大速度
- 有球员碰撞半径

球员之间不允许位置重合。

### 球和人

为了避免“穿模”和“瞬移式争球”，当前做了：

- 球自由运动时使用 `ball_motion_substeps`
- 每个子步都检查球和球员的碰撞
- 球进入球员碰撞半径后，会被推出去并对速度做反射/阻尼

因此当前满足：

- 球不会穿进球员中心
- 球不会一步穿过球员

## 比赛规则

当前支持：

- `1v1`
- `2v2`

比赛终止条件：

- 任意一方进球
- 或达到 `max_steps`

当前双边都可进球，左右球门都已可视化。

进球判定：

- 使用 `goal_half_width`
- 按球半径触线判定，而不是球心完全过线

## 争球与丢球

当前已经加入“条件触发的丢球机制”，用于抑制高频争球导致的策略坍缩。

触发位置：

- `pass`
- `trap`

风险来源：

- 附近有对手争抢
- 球速过高还强行处理球
- 上一步动作切换过猛
- 刚发生球员碰撞

失败后果：

- `loose_ball`
  球沿扰动方向弹开
- `dead_ball`
  球在附近变成低速/近静止球

设计目的：

- 不让高层学到“只要贴球就能完美处理”
- 给争球和高频切换引入真实代价

## 可视化

当前提供服务端直接导出 MP4 的能力，不依赖远程桌面。

运行：

```bash
python3 scripts/render_hypergym_episode.py
```

当前会导出：

- `videos/hypergym_demo.mp4`
- `videos/hypergym_match_demo.mp4`
- `videos/hypergym_match_2v2_demo.mp4`

渲染内容包括：

- 球场
- 双边球门
- 所有球员
- 球的位置
- 每步 action / away action / events
- step、reward、winner

## 当前 scripted policy

`policies.py` 里的 `SimpleMatchPolicy` 只是 demo 策略，不是训练目标本身。

当前特点：

- 静止自由球附近会优先尝试 `pass`
- 运动中的自由球会优先 `trap`
- 有简单的射门、穿球、补位和协防逻辑
- 支持 `1v1` 和 `2v2`

它的作用主要是：

- 让环境有可看的比赛回放
- 给后续 self-play / curriculum 提供一个初始 baseline

## 当前已知局限

当前 `hyperGym` 仍然是简化原型，不等价于真实中层控制或 Isaac Gym 物理。

主要局限：

- 没有真实脚部接触和关节控制
- 没有 3D 物理、滚动旋转、地面摩擦细节
- `dribble` 目前没有启用
- `pass` 仍然过于理想，只是比“吸附控球”保守一些
- 还没有出界规则
- 还没有“处理球朝向约束”或“处理球冷却时间”

## 推荐下一步

如果继续降低策略坍缩和 sim-to-sim gap，优先级建议是：

1. 给 `pass / trap` 增加朝向约束
2. 给处理球增加短冷却时间
3. 给连续失败加入短暂失衡窗口
4. 再考虑是否引入出界规则

当前不建议优先做：

- 过早引入复杂出界惩罚
- 恢复“碰球即吸附控球”
- 在没有真实中层能力支撑时开启理想化 `dribble`
