# VLM Zero-Training Experiments

## Goal

验证零训练视觉语言模型在 `hyperGym` 中是否能给出对人类来说可用的高层战术建议，而不是只验证接口能否跑通。

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

目的：判断模型在典型局面下的单步战术建议是否像人。

命令示例：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode benchmark \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/benchmark_qwen3vl \
  --save-json logs/vlm_poc/benchmark_qwen3vl.json
```

当前默认 benchmark case 覆盖：

- 开局自由球
- 慢速自由球停球
- 对手更接近自由球
- 中场稳控球推进
- 门前进攻
- 2v2 控球后观察空位

人工审核重点：

- `policy_id` 是否符合当前局面
- `target` 是否像一个有意义的战术目标
- 是否出现明显语义错位，例如把 `pass_to_target` 指向当前球位置

### 2. Short rollout

目的：判断模型的建议是否能连续多步维持基本足球直觉。

命令示例：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode rollout \
  --episodes 3 \
  --max-steps 40 \
  --seed 7 \
  --num-home 1 \
  --num-away 1 \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/rollout_qwen3vl \
  --save-json logs/vlm_poc/rollout_qwen3vl.json
```

人工审核重点：

- 是否反复抖动在 `move/trap/pass` 之间
- 是否经常给出落后于球位置的目标点
- `fallback_steps` 是否过高
- 多步意图是否连续，例如先拿球再推进，而不是无意义来回切换

如果默认开局大量时间都花在抢自由球，可以直接从固定 benchmark 状态启动 rollout：

```bash
OPENAI_API_KEY=... python3 scripts/vlm_policy_poc.py \
  --mode rollout \
  --episodes 2 \
  --max-steps 20 \
  --start-case-id home_midfield_possession \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/rollout_possession_qwen3vl \
  --save-json logs/vlm_poc/rollout_possession_qwen3vl.json
```

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

只有同时满足下面几点，才能认为“零训练 VLM 给出的战术建议可用”：

- 单步 benchmark 中，大多数 case 的 `policy_id` 与场景匹配
- `target` 有明确战术意义，而不是形式合法但内容空洞
- 短 rollout 中建议具备连续性，fallback 不占主导
- 再之后整局结果至少不比简单启发式差很多

## Current Findings

当前这一轮实验的直接结果：

- benchmark v1：`4/6` 命中
  - 失败模式：我方已控球时仍过度推荐 `trap_ball`
- benchmark v2：`6/6` 命中
  - 通过更强的策略语义约束，模型在控球场景下开始给出 `dribble_to_target`
- 默认开局 rollout：短期内几乎全是 `trap_ball`
  - 说明默认 reset 更像“自由球争夺测试”，不足以暴露进攻组织能力
- 从 `home_midfield_possession` 启动的 rollout：
  - 大多数步骤为连续 `dribble_to_target`
  - fallback 很低
  - 说明模型在“已控球继续推进”这个子问题上有一定零训练直觉

当前更稳妥的结论是：

- 零训练 VLM 已经能在固定单步和控球起手场景中给出基本可用的高层战术建议
- 但默认开局整局实验仍然会被“抢自由球”阶段主导，所以不能直接把当前成功率当作最终能力结论
