# Mid-Level Policy 交接文档

本文档面向接手 `mid-level / high-level policy` 方向的同学，目标是帮助新同学尽快：
- 理解当前仓库里 `pass / chase / trap` 的分工和实现方式
- 知道从哪里启动训练、看日志、定位问题
- 明确接下来最重要的研发目标：做出更好的 `pass`，并把 `trap` 和 `dribble` 做出来

---

## 1. 项目定位

当前仓库的整体思路是分层控制：
- **Low-level**：负责 locomotion / joint control / 真实物理接触，主要基于 Isaac Gym
- **Mid-level / High-level**：负责任务决策，例如追球、传球、截停、跑位、朝哪里移动

目前 mid-level 相关任务主要有：
- `ChaseBall`：更接近“追球到位”的基础任务
- `PassBall`：当前最重要的主线任务，目标是学出更稳定的传球策略
- `TrapBall`：已经有环境骨架，但还没有完整打磨成稳定训练任务
- `Dribble`：还没开始做，需要后续补一个新任务
---

## 2. 当前最重要的交付目标

接手后最重要的事情有三件：
- **把 `PassBall` 做得更好**：当前成功率还不够稳定，训练后期会退化
- **把 `TrapBall` 真正做成可训练、可评估的任务**
- **补出 `Dribble` 任务**

推荐优先级：
1. 先稳定 `PassBall`
2. 再做 `TrapBall`
3. 最后做 `Dribble`

原因：
- `PassBall` 已经是当前代码和实验最成熟的主线
- `TrapBall` 可以复用很多设计，但还缺一轮完整闭环
- `Dribble` 目前没有现成主线，建议等 `PassBall / TrapBall` 的抽象接口更稳定后再做

---

## 3. 代码结构速览

### 3.1 核心目录

- `scripts/train.py:1`
  - 当前默认训练入口
  - 目前写死为 `Runner(test=False, task_name="passBall")`
- `scripts/runner.py:1`
  - 核心训练循环
  - 任务切换、日志写入、HER、课程学习信息写入都在这里
- `envs/passBall/PassBallEnv.py:15`
  - `PassBall` 任务主环境
- `envs/passBall/PassBallEnv.yaml:1`
  - `PassBall` 任务配置
- `envs/trapBall/TrapBallEnv.py:15`
  - `TrapBall` 环境骨架
- `envs/trapBall/TrapBallEnv.yaml:1`
  - `TrapBall` 配置
- `envs/chaseBall/ChaseBallEnv.py:15`
  - `ChaseBall` 环境
- `envs/components/curriculum.py:1`
  - 课程学习逻辑
- `envs/components/ballWorld.py:1`
  - 球的重置和世界中的位置逻辑
- `eval/evaluator.py:1`
  - 评估框架 但这部分代码已经很久没有维护过了
- `eval/passBallEvaluator.py:1`
  - `PassBall` 评估逻辑
- `eval/trapBallEvaluator.py:1`
  - `TrapBall` 评估逻辑
- `logs/offline_tb_evaluator.py:1`
  - 离线读取 TensorBoard event 的轻量分析脚本
- `logs/export_tb_event_to_csv.py:1`
  - 把 tensorboard event 导出成 CSV

### 3.2 远程连接方式

建议优先使用 **Tailscale** 连接训练机，而不是直接暴露公网 SSH 端口。原因：
- 配置简单
- 不需要单独折腾复杂的内网穿透
- 在多设备之间切换比较方便
- 更适合长期维护和多人协作

推荐使用方式如下：
- 在本地电脑上安装 Tailscale
- 两边登录到同一个 tailnet（同一个账号体系或同一个团队网络）
- 服务器上线后，用 `tailscale status` 查看分配到的内网 IP 或机器名
- 本地直接通过 Tailscale 地址连接：
  - `ssh ubuntu@<tailscale-ip>`
  - 或 `ssh ubuntu@<machine-name>`

一个典型流程是：
1. 服务器安装并登录 Tailscale
2. 本地电脑安装并登录 Tailscale
3. 在 Tailscale 管理页面确认两台机器都在线
4. 本地执行 SSH 连接
5. 连上后进入仓库目录：`cd /home/ubuntu/jrWork/booster_gym`

常用命令：
- 查看本机状态：`tailscale status`
- 查看本机 IP：`tailscale ip -4`
- 如果 SSH 已启用，也可以直接：`tailscale ssh ubuntu@<machine-name>`

补充说明：
- 如果只是看训练日志，最常用的是 SSH + `tmux`
- 如果要看 TensorBoard，可以在服务器上启动后，通过 Tailscale IP 转发访问
- 首次接手时，建议先确认：
  - 训练机能否通过 Tailscale 连上
  - SSH 密钥是否已配置
  - TensorBoard 服务是否已经在后台运行


---

## 4. 训练入口与工作流

### 4.1 默认训练入口

当前默认训练入口是：
- `scripts/train.py:1`

默认行为：
- 导入 Isaac Gym
- 创建 `Runner`
- 运行 `runner.passBall()`

也就是说，现在直接执行训练脚本时，默认是在跑 `PassBall` 主线。

> python scripts/train.py --checkpoint -1 --task=PassBallEnv

这部分我写的比较差 修改训练任务时需要同时修改--task和train.py里写死的任务名称

### 4.2 核心训练逻辑

主要逻辑在：
- `scripts/runner.py:1`

需要重点关注这些内容：
- 任务初始化
- high-level agent 构建
- replay buffer 交互
- HER 注入逻辑
- TensorBoard 日志写入
- episode 成功率统计
- curriculum 更新和记录

### 4.3 常看的日志项

`scripts/runner.py` 当前会把以下关键信息打到 TensorBoard：
- `episode/success_rate`
- `episode/success`
- `episode/return`
- `episode/length`
- `curr/r_max`
- `curr/rate_global`
- `curr/rate_curr`
- `curr/changed`

对于 `PassBall`，最重要的是：
- `episode/success_rate`
- `curr/r_max`
- `events/success`
- `events/hit`

---

## 5. PassBall 的核心机制

### 5.1 当前任务定义

`PassBall` 的目标不是“复杂带球”，而是让 mid-level policy 学会：
- 接近球
- 组织朝目标方向的动作
- 让球进入 target 附近，视作成功

核心环境文件：
- `envs/passBall/PassBallEnv.py:15`

### 5.2 关键设计：球固定，target 变难

`PassBall` 当前不是通过“把球刷得更远”来增加难度，而是：
- 球固定生成在机器人前方
- target 的位置按 curriculum 动态采样

对应代码：
- 球固定生成：`envs/components/ballWorld.py:90`
- target 采样：`scripts/runner.py:97`

当前机制是：
- 球固定在机器人前方约 `0.6m`
- target 在机器人前方 ±30° 扇形里采样
- 半径范围由课程学习窗口 `[cur_r_min, cur_r_max]` 决定

### 5.3 课程学习

课程学习逻辑在：
- `envs/components/curriculum.py:1`

当前 `PassBall` 配置在：
- `envs/passBall/PassBallEnv.yaml:279`

重要参数：
- `r_min`
- `r_max`
- `inc`
- `dec`
- `high_thresh`
- `low_thresh`

当前语义是：
- 成功率高于 `0.6` 才升难
- 成功率低于 `0.3` 才降难
- 落在中间区间时，难度保持不变


### 5.4 当前最主要问题

基于最近一批实验，`PassBall` 目前的典型问题是：
- 前期成功率上升较快
- 训练时间拉长后，成功率会回落到 `0.4 ~ 0.5`
- curriculum 会停在一个中间难度，不再继续升，也不主动降

当前观察到的结论是：
- `PassBallEnv_20260309-201202`：长训到 50w+ step 后，成功率大致在 `0.44 ~ 0.47`

现象解释：
- 训练后期策略没有继续提高
- 但又没有差到低于降难阈值
- 结果就是卡在一个中等难度层反复震荡

### 5.5 HER

`PassBall` 当前有 HER 逻辑：
- 位置：`scripts/runner.py:752`
- 环境重算 reward 的接口：`envs/passBall/PassBallEnv.py:723`

HER 的作用：
- 当 episode 没成功但碰到了球时，采样球最终位置作为虚拟 goal
- 让高层策略多得到一些“成功型回放”

注意：
- 当前 HER 只是一个增强项，不是主问题的根治手段
- 如果 success rate 长期卡在 40% 左右，优先查 reward / obs / curriculum / action space，而不是先继续堆 HER

---

## 6. TrapBall 的现状

### 6.1 代码位置

- `envs/trapBall/TrapBallEnv.py:15`
- `envs/trapBall/TrapBallEnv.yaml:1`
- `eval/trapBallEvaluator.py:1`

### 6.2 当前状态

`TrapBall` 已经有环境文件和 evaluator，但目前属于“骨架已在，闭环未充分验证”的状态。

建议接手同学先确认下面几件事：
- 训练入口是否已完整接到 `Runner`
- reward 是否足够稳定
- success 定义是否清晰、是否和 evaluator 一致
- TensorBoard 日志字段是否完整

### 6.3 推荐推进方式

建议按 `PassBall` 的经验复制一版最小闭环：
1. 先保证 task 能稳定 reset / terminate
2. 再保证 success 定义清楚
3. 再补 reward shaping
4. 最后再考虑 curriculum
---

## 7. 评估与分析方法

### 7.1 在线评估

训练时主要看 TensorBoard：
- `episode/success_rate`
- `episode/success`
- `episode/return`
- `curr/r_max`

### 7.2 离线分析脚本

建议优先用这两个脚本：
- `logs/offline_tb_evaluator.py:1`
- `logs/export_tb_event_to_csv.py:1`

用途：
- 大 event 文件下做轻量离线分析
- 导出 success rate / curriculum / reward 曲线
