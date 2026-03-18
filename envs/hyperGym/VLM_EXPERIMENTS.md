# VLM Zero-Training Experiments

## Goal

验证零训练视觉语言模型在 `hyperGym` 中是否能给出对人类来说可用的高层战术建议，而不是只验证接口能否跑通。

当前主输入模态有两种：

- `image_with_state_text`
  - 图像 + 标准状态文本摘要
- `image_only`
  - 只给图像，不给状态文本

建议后续所有结论都按这两个 observation mode 分开记录，避免把输入模态差异混进同一条结论里。

当前视觉视角支持两类：

- `global`
  - 原始全局俯视图
- `ego_<player_id>`
  - 以某个球员为中心的局部俯视裁剪图
  - 例如 `ego_home_0`
- `ego_fp_<player_id>`
  - 以某个球员为视角中心的第一人称局部视图
  - 球员初始默认朝向对方球门
  - 后续朝向优先跟随球员速度方向
  - 例如 `ego_fp_home_0`

建议把“输入模态”和“视觉视角”视为两条独立消融轴，分别记录。

## Worklist

- [x] 把 VLM PoC 保持为独立实验入口，不接主训练流程
- [x] 加入逐步 artifact 导出，保存每步图像、状态摘要、原始输出、解析后动作和环境反馈
- [x] 加入固定状态 benchmark，避免只靠整局成功率判断模型质量
- [x] 收紧 prompt 和 schema 约束，减少无意义 `pass_to_target`
- [x] 给实验流程写成可重复执行的文档
- [x] 运行 benchmark 模式，人工审阅每个 case 的策略建议是否符合直觉
- [x] 运行短 rollout 模式，检查建议的连续性、fallback 比例和明显反直觉动作
- [x] 运行从固定控球状态启动的 rollout，避免默认开局一直停留在抢自由球阶段
- [x] 若 benchmark 结果不稳定，继续迭代 prompt 或输出约束
- [x] 若 benchmark 结果稳定，再扩大到多 seed 整局实验

如果后续发现新的必要任务，应直接追加到这份工作列表。

## Evaluation Stages

### 1. Single-step benchmark

目的：判断模型在典型局面下，给每个 `home` 球员分配的战术建议是否像人。

命令示例：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode benchmark \
  --observation-mode image_with_state_text \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/benchmark_qwen3vl \
  --save-json logs/vlm_poc/benchmark_qwen3vl.json
```

只看图像的对照组：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode benchmark \
  --observation-mode image_only \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/benchmark_qwen3vl_image_only \
  --save-json logs/vlm_poc/benchmark_qwen3vl_image_only.json
```

当前默认 benchmark case 覆盖：

- 开局自由球
- 慢速自由球停球
- 对手更接近自由球
- 中场稳控球推进
- 门前进攻
- 2v2 控球后观察空位

人工审核重点：

- 主控球球员的 `policy_id` 是否符合当前局面
- 无球队友是否给出了合理的支援或补位动作
- `target` 是否像一个有意义的战术目标
- 是否出现明显语义错位，例如把 `pass_to_target` 指向当前球位置

### 2. Short rollout

目的：判断模型给整支 `home` 队的建议是否能连续多步维持基本足球直觉。

命令示例：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode rollout \
  --episodes 3 \
  --max-steps 40 \
  --seed 7 \
  --num-home 1 \
  --num-away 1 \
  --observation-mode image_with_state_text \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/rollout_qwen3vl \
  --save-json logs/vlm_poc/rollout_qwen3vl.json
```

人工审核重点：

- 主控球球员是否反复抖动在 `move/trap/pass` 之间
- 无球队友是否长期原地不动，或跑位完全无意义
- 是否经常给出落后于球位置的目标点
- `fallback_steps` 是否过高
- 多步意图是否连续，例如一名球员先拿球再推进，其他球员同步形成支援，而不是无意义来回切换

如果默认开局大量时间都花在抢自由球，可以直接从固定 benchmark 状态启动 rollout：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode rollout \
  --episodes 2 \
  --max-steps 20 \
  --start-case-id home_midfield_possession \
  --observation-mode image_with_state_text \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/rollout_possession_qwen3vl \
  --save-json logs/vlm_poc/rollout_possession_qwen3vl.json
```

只看图像的控球起手对照组：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode rollout \
  --episodes 2 \
  --max-steps 20 \
  --start-case-id home_midfield_possession \
  --observation-mode image_only \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/rollout_possession_qwen3vl_image_only \
  --save-json logs/vlm_poc/rollout_possession_qwen3vl_image_only.json
```

### 2.5 Oracle correctness ablation

目的：直接测量去掉状态文本后，VLM 和 scripted oracle 的差距会扩大多少。

基线：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_oracle_correctness_eval.py \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_with_state_text \
  --vision-view global \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl.json
```

对照组：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_oracle_correctness_eval.py \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_only \
  --vision-view global \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_image_only \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_image_only.json
```

### 2.6 Ego-centric image ablation

目的：只改图像视角，不改环境动力学，测试局部 ego-centric 视觉下 VLM 的策略质量。

推荐先固定：

- `episode_seeds = 7,8,9`
- `observation_mode = image_with_state_text`
- `vision_view = ego_fp_home_0`

命令示例：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_oracle_correctness_eval.py \
  --episode-seeds 7,8,9 \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_with_state_text \
  --vision-view ego_fp_home_0 \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_seed789 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_seed789.json
```

如果要看“只给 ego-centric 图像、不再给状态文本”的更激进设置：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_oracle_correctness_eval.py \
  --episode-seeds 7,8,9 \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_only \
  --vision-view ego_fp_home_0 \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_image_only_seed789 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_image_only_seed789.json
```

### 2.7 Visual-boundary benchmark cases

目的：构造“全局图和第一人称图应该产生不同判断”的单步 case，避免 oracle trajectory 采样帧本身过于简单。

当前新增的视觉敏感 case：

- `visual_blindside_recycle`
  - 安全出球点在 `home_0` 身后左侧
- `visual_far_side_switch`
  - 远侧换边机会只在全局图里清楚
- `visual_back_post_runner`
  - 门前远门柱跑位需要全局空间关系
- `visual_trailing_press_warning`
  - 身后压迫风险在第一人称里更难感知

推荐先跑这两组对照：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode benchmark \
  --observation-mode image_only \
  --vision-view global \
  --case-ids visual_blindside_recycle,visual_far_side_switch,visual_back_post_runner,visual_trailing_press_warning \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/visual_boundary_global_image_only \
  --save-json logs/vlm_poc/visual_boundary_global_image_only.json
```

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode benchmark \
  --observation-mode image_only \
  --vision-view ego_fp_home_0 \
  --case-ids visual_blindside_recycle,visual_far_side_switch,visual_back_post_runner,visual_trailing_press_warning \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/visual_boundary_ego_fp_home_0_image_only \
  --save-json logs/vlm_poc/visual_boundary_ego_fp_home_0_image_only.json
```

人工审核重点：

- `home_0` 是否还能发现全局图里可见、但第一人称下难见的支援点
- 第一人称下是否更容易继续前带、而不是做换边或回做
- 身后压迫是否只在全局图里被显式利用

### 3. Full rollout

目的：在单步质量和短滚动都可接受后，再看整局胜率和终局表现。

不建议在前两步没过之前就把成功率当主结论。

## Artifact Layout

如果传入 `--save-artifacts-dir`，脚本会保存：

- `*.png`
  当前步骤或 benchmark case 的输入帧
- `*.json`
  对应的状态摘要、原始 VLM 输出、解析后动作、环境反馈

rollout 模式下，artifact 会按 `episode_000/ep000_step000.*` 组织。

## Current Interpretation Standard

只有同时满足下面几点，才能认为“零训练 VLM 给出的多人战术建议可用”：

- 单步 benchmark 中，主控球球员和无球队友的大多数动作都与场景匹配
- `target` 有明确战术意义，而不是形式合法但内容空洞
- 短 rollout 中建议具备连续性，fallback 不占主导
- 再之后整局结果至少不比简单启发式差很多

## Current Findings

当前这一轮实验的直接结果：

- benchmark v1：`4/6` 命中
  - 失败模式：主控球球员在已控球时仍过度推荐 `trap_ball`
- benchmark v2：`6/6` 命中
  - 通过更强的策略语义约束，模型在控球场景下开始给出 `dribble_to_target`，并给无球队友输出支援跑位
- 默认开局 rollout：短期内几乎全是 `trap_ball`
  - 说明默认 reset 更像“自由球争夺测试”，不足以暴露多人进攻组织能力
- 从 `home_midfield_possession` 启动的 rollout：
  - 主控球球员大多数步骤为连续 `dribble_to_target`
  - 无球队友能给出 `move_to_target` 形式的支援动作
  - fallback 很低
  - 说明模型在“已控球继续推进 + 队友支援跑位”这个子问题上有一定零训练直觉

当前更稳妥的结论是：

- 零训练 VLM 已经能在固定单步和控球起手场景中给出基本可用的多人高层战术建议
- 但默认开局整局实验仍然会被“抢自由球”阶段主导，所以不能直接把当前成功率当作最终能力结论
