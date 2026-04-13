# VLM Ablation Results

本文件单独记录 VLM 输入条件的对比实验结果。

后续所有消融默认复用同一组固定 oracle 轨迹 seed，避免不同条件之间抽到不同轨迹。

## Summary Matrix

### 一般情况

oracle correctness 采样帧，可视为一般情况。

| 观察 | 状态数值文本 + BEV | 无状态 + BEV | 状态数值文本 + ego | 无状态 + ego |
| --- | --- | --- | --- | --- |
| 去掉状态文本的影响 | 基线，高，oracle correctness 表现好 | 相对 BEV 基线几乎无明显下降 | 基线，与对应 BEV 几乎一致 | 相对 ego 基线几乎无明显下降 |
| 视角变化（BEV → ego） | BEV 基线，高 | BEV 基线，高 | 相对对应 BEV 基线几乎无明显变化 | 相对对应 BEV 基线几乎无明显变化 |

### 困难情况

专门构造的困难视觉 case，对 ego 视角尤其困难。

| 观察 | 状态数值文本 + BEV | 无状态 + BEV | 状态数值文本 + ego | 无状态 + ego |
| --- | --- | --- | --- | --- |
| 去掉状态文本的影响 | 基线即明显暴露局限，4 个 case 基本都未做出期望传球 | 相对有状态基线未观察到进一步恶化，同样失败 | 基线即明显暴露局限，4 个 case 基本都未做出期望传球 | 相对有状态基线未观察到进一步恶化，同样失败 |
| 视角差异（BEV vs ego） | BEV 基线已失败 | BEV 基线已失败 | 相对对应 BEV 基线未观察到进一步恶化，同样失败 | 相对对应 BEV 基线未观察到进一步恶化，同样失败 |

### 统计型 Scenario Suite

混合 easy / medium / hard 起始场景的小型统计评测，当前是最接近你想要的“看总体偏差量”的设置。

| 观察 | 状态数值文本 + BEV | 无状态 + BEV | 状态数值文本 + ego | 无状态 + ego |
| --- | --- | --- | --- | --- |
| 去掉状态文本的影响 | 基线，`policy_accuracy=0.7111`，`step_exact_match_rate=0.5556` | 相对 BEV 基线下降到 `0.5333 / 0.3333` | 基线，`policy_accuracy=0.7111`，`step_exact_match_rate=0.5556` | 相对 ego 基线下降到 `0.6000 / 0.4074` |
| 视角差异（BEV vs ego） | BEV 基线，`0.7111 / 0.5556` | BEV 基线，`0.5333 / 0.3333` | 与对应 BEV 基线几乎相同 | 相对对应 BEV 基线略好，但差异小于“有无状态文本”的差异 |
| 当前解读 | 当前最支持“状态文本比视角变化更重要” | 同左 | 同左 | 同左，但这一轮仍受较高 fallback/timeout 污染，只能视为方向性证据 |

### 图像源对照

在同一批 oracle 采样帧、同一份状态文本、同一份 scripted oracle 标签下，只替换图像来源：

- `HyperGym BEV map`
- `IsaacGym top-down RGB`

| 观察 | HyperGym BEV + 状态数值文本 | IsaacGym top-down + 状态数值文本 |
| --- | --- | --- |
| 相对 scripted oracle 的正确性 | `policy_accuracy=0.8611`，`step_exact_match_rate=0.7222` | `policy_accuracy=0.9167`，`step_exact_match_rate=0.8333` |
| 目标点误差 | `target_hit_rate=0.7222`，`mean_target_error=0.6221` | `target_hit_rate=0.6111`，`mean_target_error=0.8701` |
| VLM 输出本身是否变化 | 基线 | 相对 HyperGym 图像有明显变化：`16/18` 个样本、`19/36` 个球员预测发生变化 |
| 当前解读 | HyperGym 图像下，VLM 更接近原先那条 oracle correctness 评测链 | Isaac 图像会显著改变 VLM 的策略输出；在这批样本上它反而提高了动作类别匹配率，但目标点更偏 |

当前实验表明，在一般 oracle 采样帧上，去掉状态文本和把 BEV 换成 ego 视角都几乎不影响结果；但在专门构造的困难视觉 case 上，一旦不给状态文本，无论 BEV 还是 ego 都明显失败。也就是说，当前主要瓶颈不是 ego 视角本身，而是 VLM 在纯图像条件下处理高难战术空间关系的能力不足。

- 记录时间：2026-03-17 17:20:10 CST
- 环境：`hyperGym`
- 评测脚本：`scripts/tmp/vlm_oracle_correctness_eval.py`
- 模型：`qwen3vl`
- 接口：`https://models.sjtu.edu.cn/api/v1`

## Ablation 001

### Question

如果去掉标准状态文本，只给图像，VLM 相对 scripted oracle 的正确性会不会下降？

### Compared Conditions

- `image_with_state_text`
  - 图像 + 标准状态文本摘要
- `image_only`
  - 只给图像，不给状态文本

### Shared Evaluation Setup

- 固定轨迹 seed：`[7, 8, 9]`
- `episodes = 3`
- `max_steps = 30`
- `samples_per_episode = 6`
- `num_home = 2`
- `num_away = 2`
- 总样本数：`18`
- 总球员动作预测数：`36`
- 汇总方式：
  - 先在每条轨迹上统计指标
  - 再对三条轨迹取平均

### Commands

```bash
OPENAI_API_KEY=*** python3 scripts/tmp/vlm_oracle_correctness_eval.py \
  --episode-seeds 7,8,9 \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_with_state_text \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_state_text_seed789 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_state_text_seed789.json
```

```bash
OPENAI_API_KEY=*** python3 scripts/tmp/vlm_oracle_correctness_eval.py \
  --episode-seeds 7,8,9 \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_only \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_image_only_seed789 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_image_only_seed789.json
```

### Result Summary


| Condition               | policy_accuracy | step_exact_match_rate | target_hit_rate | mean_target_error | median_target_error |
| ----------------------- | --------------- | --------------------- | --------------- | ----------------- | ------------------- |
| `image_with_state_text` | `0.9722`        | `0.9444`              | `0.5000`        | `1.4855`          | `0.6475`            |
| `image_only`            | `0.9722`        | `0.9444`              | `0.5000`        | `1.4855`          | `0.6475`            |


这里的数值同时等于：

- 三条轨迹的宏平均
- 所有样本合并后的整体统计

因为三条轨迹的样本数相同，都是 `6` 帧。

### Per-trajectory Breakdown

两组条件的逐轨迹结果也完全一致：


| episode_seed | policy_accuracy | step_exact_match_rate | target_hit_rate | mean_target_error | median_target_error |
| ------------ | --------------- | --------------------- | --------------- | ----------------- | ------------------- |
| `7`          | `1.0000`        | `1.0000`              | `0.5000`        | `1.2304`          | `0.8214`            |
| `8`          | `0.9167`        | `0.8333`              | `0.4167`        | `2.2298`          | `1.0849`            |
| `9`          | `1.0000`        | `1.0000`              | `0.5833`        | `0.9964`          | `0.2901`            |


### Per-policy Confusion

两组结果完全一致：

- `move_to_target -> move_to_target`: `18`
- `trap_ball -> trap_ball`: `17`
- `pass_to_target -> trap_ball`: `1`

### Direct Comparison

- 两组三轨迹平均指标完全相同
- 两组三条轨迹的逐轨迹指标也完全相同
- `18/18` 个样本逐样本结果完全一致
- `36/36` 个球员预测逐球员完全一致
- 不只是 accuracy 一样，连 `pred_policy_id`、`pred_target`、`pred_reason` 都完全一致

### Interpretation

在固定三条 oracle 轨迹 `seed = [7, 8, 9]` 上，没有观测到“去掉状态文本导致性能下降”。

更具体地说，本次实验支持下面这个结论：

- 对当前这批 `2v2` 采样帧，`qwen3vl` 的输出对是否提供标准状态文本不敏感

但这个结果还不能直接推出更强的结论，例如：

- 模型从来不需要状态文本
- 文本输入对所有局面都没有帮助

当前更稳妥的解释有三种：

- 这些采样帧里的关键信息已经能从图像中直接恢复
- 当前状态文本提供的信息，没有改变模型在这批样本上的决策边界
- 当前 prompt 里文本虽然存在，但没有在这批局面上产生额外约束作用

### Additional Note

仓库里旧的基线结果 [logs/vlm_poc/oracle_correctness_summary.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_summary.json) 与本次新跑的 `image_with_state_text` 数值不同：

- 旧结果：`policy_accuracy = 0.8333`
- 新结果：`policy_accuracy = 0.9722`

同时新结果的 target 指标更差：

- 旧结果：`target_hit_rate = 0.6389`, `mean_target_error = 1.2773`
- 新结果：`target_hit_rate = 0.5000`, `mean_target_error = 1.4855`

这说明除了“是否给状态文本”之外，模型后端或服务端行为本身也可能发生了漂移。因此后续做条件对比时，应尽量把所有对照组放在同一时间窗口内一起跑，而不要跨天直接比较绝对数值。

另外，后续消融建议统一复用：

- `episode_seeds = [7, 8, 9]`

不要只记录一个 `base seed`，否则无法保证比较的是同一组三条轨迹。

### Artifacts

- 基线结果 JSON: [logs/vlm_poc/oracle_correctness_qwen3vl_state_text_seed789.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_state_text_seed789.json)
- 对照结果 JSON: [logs/vlm_poc/oracle_correctness_qwen3vl_image_only_seed789.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_image_only_seed789.json)
- 基线 artifact 目录: [logs/vlm_poc/oracle_correctness_qwen3vl_state_text_seed789](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_state_text_seed789)
- 对照 artifact 目录: [logs/vlm_poc/oracle_correctness_qwen3vl_image_only_seed789](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_image_only_seed789)

### Suggested Next Ablations

- 去掉 `recent_events`，只保留静态状态文本
- 去掉球速度和控球方，只保留位置文本
- 图像降采样或裁剪，测试空间关系识别的下限
- 只给文本，不给图像，测纯符号状态输入上限

## Ablation 002

### Question

把全局俯视图换成第一人称 ego 视角后，VLM 相对 scripted oracle 的正确性会不会下降？

### Compared Conditions

- `global + image_with_state_text`
  - 原始全局俯视图
  - 标准状态文本摘要
- `ego_fp_home_0 + image_with_state_text`
  - `home_0` 第一人称 ego 视角
  - 标准状态文本摘要

### Shared Evaluation Setup

- 固定轨迹 seed：`[7, 8, 9]`
- `episodes = 3`
- `max_steps = 30`
- `samples_per_episode = 6`
- `num_home = 2`
- `num_away = 2`
- 总样本数：`18`
- 总球员动作预测数：`36`

### Commands

```bash
OPENAI_API_KEY=*** python3 scripts/tmp/vlm_oracle_correctness_eval.py \
  --episode-seeds 7,8,9 \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_with_state_text \
  --vision-view global \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_global_state_text_seed789_v2 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_global_state_text_seed789_v2.json
```

```bash
OPENAI_API_KEY=*** python3 scripts/tmp/vlm_oracle_correctness_eval.py \
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
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_state_text_seed789 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_state_text_seed789.json
```

### Result Summary


| Condition                               | policy_accuracy | step_exact_match_rate | target_hit_rate | mean_target_error | median_target_error |
| --------------------------------------- | --------------- | --------------------- | --------------- | ----------------- | ------------------- |
| `global + image_with_state_text`        | `0.9722`        | `0.9444`              | `0.5000`        | `1.4855`          | `0.6475`            |
| `ego_fp_home_0 + image_with_state_text` | `0.9722`        | `0.9444`              | `0.5000`        | `1.4855`          | `0.6475`            |


### Direct Comparison

- 两组核心指标完全相同
- 两组三条轨迹的逐轨迹指标完全相同
- `18/18` 个样本逐样本结果完全一致
- `36/36` 个球员预测逐球员完全一致
- 不只是动作类型一样，连 `pred_policy_id`、`pred_target`、`pred_reason` 都完全一致

### Interpretation

在固定三条轨迹 `seed = [7, 8, 9]` 上，把图像从全局俯视图换成第一人称 ego 视角后，没有观测到性能下降。

当前最合理的解释不是“第一人称视角和全局视角等价”，而是：

- 在 `image_with_state_text` 设定下，状态文本已经提供了足够强的局面信息
- 模型的决策几乎完全由文本驱动
- 因此视觉视角变化没有进入当前这条评测链的决策边界

也就是说，这个结果更像是在说明：

- 当前带状态文本的 oracle correctness 实验，对视觉视角并不敏感

而不是在说明：

- 模型真的理解了第一人称局部视角里的空间关系

### Prompt Adjustment Note

这次没有继续修改 prompt。

原因是：

- 结果不是“视角一变就崩”，而是“完全不变”
- 在这种情况下，优先怀疑的是文本输入覆盖了视觉信息，而不是 prompt 对第一人称图像解释不够

如果后续要更真实地测第一人称视觉能力，下一步更值得跑的是：

- `ego_fp_home_0 + image_only`
- `global + image_only`

只有在去掉状态文本之后，视觉视角差异才更可能真正显现出来。

### Artifacts

- 全局结果 JSON: [logs/vlm_poc/oracle_correctness_qwen3vl_global_state_text_seed789_v2.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_global_state_text_seed789_v2.json)
- 第一人称结果 JSON: [logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_state_text_seed789.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_state_text_seed789.json)
- 全局 artifact 目录: [logs/vlm_poc/oracle_correctness_qwen3vl_global_state_text_seed789_v2](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_global_state_text_seed789_v2)
- 第一人称 artifact 目录: [logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_state_text_seed789](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_state_text_seed789)

## Ablation 003

### Question

在第一人称 ego 视角下，如果进一步去掉状态文本，VLM 的正确性会不会下降？

### Compared Conditions

- `global + image_only`
  - 全局俯视图
  - 不给状态文本
- `ego_fp_home_0 + image_only`
  - `home_0` 第一人称 ego 视角
  - 不给状态文本

### Shared Evaluation Setup

- 固定轨迹 seed：`[7, 8, 9]`
- `episodes = 3`
- `max_steps = 30`
- `samples_per_episode = 6`
- `num_home = 2`
- `num_away = 2`
- 总样本数：`18`
- 总球员动作预测数：`36`

### Commands

```bash
OPENAI_API_KEY=*** python3 scripts/tmp/vlm_oracle_correctness_eval.py \
  --episode-seeds 7,8,9 \
  --episodes 3 \
  --max-steps 30 \
  --samples-per-episode 6 \
  --num-home 2 \
  --num-away 2 \
  --observation-mode image_only \
  --vision-view global \
  --vlm-model qwen3vl \
  --vlm-base-url https://models.sjtu.edu.cn/api/v1 \
  --save-artifacts-dir logs/vlm_poc/oracle_correctness_qwen3vl_global_image_only_seed789_v2 \
  --save-json logs/vlm_poc/oracle_correctness_qwen3vl_global_image_only_seed789_v2.json
```

```bash
OPENAI_API_KEY=*** python3 scripts/tmp/vlm_oracle_correctness_eval.py \
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

### Result Summary


| Condition                    | policy_accuracy | step_exact_match_rate | target_hit_rate | mean_target_error | median_target_error |
| ---------------------------- | --------------- | --------------------- | --------------- | ----------------- | ------------------- |
| `global + image_only`        | `0.9722`        | `0.9444`              | `0.5000`        | `1.4855`          | `0.6475`            |
| `ego_fp_home_0 + image_only` | `0.9722`        | `0.9444`              | `0.5000`        | `1.4855`          | `0.6475`            |


### Direct Comparison

- 两组核心指标完全相同
- `18/18` 个样本逐样本结果完全一致
- `36/36` 个球员预测逐球员完全一致
- 连 `pred_policy_id`、`pred_target`、`pred_reason` 都完全一致

### Implementation Check

图像本身确实不同，不是实现 bug 导致发了同一张图：

- `ep000_step001` 的 `global` 与 `ego_fp` PNG 哈希不同
- `ep001_step001` 的 `global` 与 `ego_fp` PNG 哈希也不同

也就是说：

- 送进模型的图像视角确实变了
- 但模型输出仍然完全不变

### Interpretation

这次结果比前两组更强，因为它说明：

- 不只是“带状态文本时视角变化无效”
- 连在 `image_only` 设定下，`global` 和 `ego_fp_home_0` 的视觉差异也没有反映到输出上

在当前这批 oracle 样本上，更稳妥的解释是：

- 模型对这些样本的决策主要由非常稳定的先验模式驱动
- 当前样本本身可能过于简单，导致不同图像都映射到同一套高层策略
- 或者模型虽然看到了图像，但这批帧里视觉差异没有影响它的动作选择

目前仍然不能直接证明：

- 模型完全不看图像

但可以明确说明：

- 在当前 oracle correctness 采样协议下，图像视角变化没有成为有效变量

### What This Means For Next Experiments

如果后续要继续追视觉能力边界，优先级更高的不是继续改 prompt，而是改样本协议：

- 增加专门为视觉歧义设计的 benchmark case
- 增加只靠图像才能区分的局面
- 增加遮挡、局部视野缺失、远距离空位判断这类场景

否则继续在当前这批 oracle 采样帧上换视角，信息量可能已经很有限了。

### Artifacts

- 全局结果 JSON: [logs/vlm_poc/oracle_correctness_qwen3vl_global_image_only_seed789_v2.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_global_image_only_seed789_v2.json)
- 第一人称结果 JSON: [logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_image_only_seed789.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_image_only_seed789.json)
- 全局 artifact 目录: [logs/vlm_poc/oracle_correctness_qwen3vl_global_image_only_seed789_v2](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_global_image_only_seed789_v2)
- 第一人称 artifact 目录: [logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_image_only_seed789](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_correctness_qwen3vl_ego_fp_home_0_image_only_seed789)

## Ablation 004

### Question

如果不用 oracle trajectory 采样帧，而是改用专门构造的视觉敏感单步 case，`global` 和 `ego_fp_home_0` 会不会开始分叉？

### Compared Conditions

- `global + image_only`
- `ego_fp_home_0 + image_only`

### Benchmark Cases

本组只跑 4 个专门构造的视觉边界 case：

- `visual_blindside_recycle`
- `visual_far_side_switch`
- `visual_back_post_runner`
- `visual_trailing_press_warning`

这些 case 的共同目标是：

- 让 `home_0` 的合理动作应该是 `pass_to_target`
- 让关键信息尽量落在第一人称盲区、弱可见区域或必须依赖全局空间关系的位置

### Prompt Note

在正式记录结果前，做过一次很小的 prompt 修正：

- 增加渲染图例说明
- 明确告诉模型：黄色球是球，黄色光环表示该球员当前控球

这个修正是必要的，因为第一次运行时模型把多个 case 误读成“自由球”，甚至输出 `trap_ball`。

修正后再跑，`trap_ball` 误判消失，但核心结论没有改变。

### Result Summary

| Condition | cases | policy_matches | policy_match_rate |
| --- | ---: | ---: | ---: |
| `global + image_only` | `4` | `0` | `0.0000` |
| `ego_fp_home_0 + image_only` | `4` | `0` | `0.0000` |

### Case-level Outputs

图例修正后的最终结果：

- `global + image_only`
  - `visual_blindside_recycle -> move_to_target`
  - `visual_far_side_switch -> move_to_target`
  - `visual_back_post_runner -> move_to_target`
  - `visual_trailing_press_warning -> move_to_target`
- `ego_fp_home_0 + image_only`
  - `visual_blindside_recycle -> move_to_target`
  - `visual_far_side_switch -> move_to_target`
  - `visual_back_post_runner -> move_to_target`
  - `visual_trailing_press_warning -> move_to_target`

### Interpretation

这组结果说明两件事：

- 专门构造视觉敏感 case 之后，模型终于不再“表面上高分”
- 但它仍然没有表现出利用全局空间关系去主动换边、回做或找远门柱跑位的能力

也就是说，新的 benchmark 确实比 oracle trajectory 采样帧更敏感，但当前 `qwen3vl` 在这些需要显式空间组织的 case 上没有通过。

更具体地看：

- 问题不只是第一人称视角太难
- 因为连 `global + image_only` 也同样是 `0/4`
- 当前失败模式更像是：模型默认退回保守推进，而不是识别并执行结构化传球

### What Changed After Prompt Fix

图例修正带来的唯一明确改进是：

- 模型不再把这些 case 误读为“自由球需要 trap”

但图例修正没有带来：

- 从 `move_to_target` 转向 `pass_to_target`

所以当前瓶颈已经不是“渲染符号没解释清楚”，而是：

- 模型没有在这些视觉边界 case 上学会主动做更激进的全局传球判断

### Artifacts

- 全局结果 JSON: [logs/vlm_poc/visual_boundary_global_image_only_v2.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_global_image_only_v2.json)
- 第一人称结果 JSON: [logs/vlm_poc/visual_boundary_ego_fp_home_0_image_only_v2.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_ego_fp_home_0_image_only_v2.json)
- 全局 artifact 目录: [logs/vlm_poc/visual_boundary_global_image_only_v2](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_global_image_only_v2)
- 第一人称 artifact 目录: [logs/vlm_poc/visual_boundary_ego_fp_home_0_image_only_v2](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_ego_fp_home_0_image_only_v2)

## Ablation 005

### Question

在同一组困难视觉 benchmark case 上，如果恢复状态文本，BEV 或 ego 视角是否会明显改善？

### Compared Conditions

- `global + image_with_state_text`
- `ego_fp_home_0 + image_with_state_text`

### Result Summary

| Condition | cases | policy_matches | policy_match_rate |
| --- | ---: | ---: | ---: |
| `global + image_with_state_text` | `4` | `0` | `0.0000` |
| `ego_fp_home_0 + image_with_state_text` | `4` | `0` | `0.0000` |

### Case-level Outputs

两组最终都是：

- `visual_blindside_recycle -> move_to_target`
- `visual_far_side_switch -> move_to_target`
- `visual_back_post_runner -> move_to_target`
- `visual_trailing_press_warning -> move_to_target`

### Interpretation

这说明在这组专门构造的困难视觉 case 上，恢复状态文本也没有把模型拉回到我们期望的 `pass_to_target` 决策。

所以当前更稳妥的结论是：

- 这些 case 暴露出来的不是“去掉状态文本后才出现的问题”
- 而是模型本身在这类高难空间组织决策上就偏保守

换句话说：

- 在一般 oracle 采样帧上，去掉状态文本几乎不影响结果
- 但在困难视觉 case 上，加回状态文本也没有救回来

这两者合在一起说明：

- 当前困难 case 更像是在测模型的战术推理上限
- 而不只是测它是否拿到了足够的状态输入

### Artifacts

- 全局结果 JSON: [logs/vlm_poc/visual_boundary_global_state_text_v1.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_global_state_text_v1.json)
- 第一人称结果 JSON: [logs/vlm_poc/visual_boundary_ego_fp_home_0_state_text_v1.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_ego_fp_home_0_state_text_v1.json)
- 全局 artifact 目录: [logs/vlm_poc/visual_boundary_global_state_text_v1](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_global_state_text_v1)
- 第一人称 artifact 目录: [logs/vlm_poc/visual_boundary_ego_fp_home_0_state_text_v1](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/visual_boundary_ego_fp_home_0_state_text_v1)

## Ablation 006

### Question

如果不用少量单步 case，而是用一个混合难度的 scenario suite 做统计型比较，`有无状态文本` 和 `BEV/ego` 能不能开始拉开差异？

### Scenario Suite

本轮使用 3 个起始场景，覆盖 easy / medium / hard：

- `home_midfield_possession`
- `medium_pressure_release`
- `visual_trailing_press_warning`

统计设置：

- `episode_seeds = [7, 8, 9]`
- 每个起始场景 × 每个 seed rollout `8` 步
- 每条 rollout 采样 `3` 步
- 总共 `9` 条 rollout
- 总共 `27` 个比较步
- 总共 `45` 个球员动作比较

### 2x2 Summary

| Setting | policy_accuracy | step_exact_match_rate | 不一致步数 / 27 |
| --- | ---: | ---: | ---: |
| `state + BEV` | `0.7111` | `0.5556` | `12` |
| `no_state + BEV` | `0.5333` | `0.3333` | `18` |
| `state + ego` | `0.7111` | `0.5556` | `12` |
| `no_state + ego` | `0.6000` | `0.4074` | `16` |

### What This Suggests

这组统计型 suite 首次开始出现你真正关心的量化信号：

- 去掉状态文本后，整体偏差变大
- 在这组 suite 上，`BEV` 和 `ego` 的差异存在，但明显小于“有无状态文本”的差异

按这轮结果，最粗略的读取是：

- 状态文本帮助明显
- 视角变化影响较小，至少在这组 suite 上没有压倒性主效应

### Per-scenario Signal

分场景看，状态文本带来的增益主要集中在：

- `home_midfield_possession`

而在另外两类更复杂的场景里：

- `medium_pressure_release`
- `visual_trailing_press_warning`

`BEV` 和 `ego` 的差异仍然不大。

### Important Caveat

这轮统计结果仍然是 provisional，只能当方向性证据，原因是网络/API 不稳定导致 fallback 比例较高：

- `state + BEV`: `39 / 45` 预测走了 fallback
- `no_state + BEV`: `23 / 45`
- `state + ego`: `28 / 45`
- `no_state + ego`: `24 / 45`

因此这轮结果虽然开始出现趋势，但还不能作为最终定量结论。

更稳妥的说法是：

- 这轮 suite 初步支持“状态文本比视角变化更重要”
- 但接口超时/fallback 污染仍然较重，后续最好在更稳定的服务窗口再复核一次

### Artifacts

- `state + BEV`: [scenario_suite_small_global_state_text.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/scenario_suite_small_global_state_text.json)
- `no_state + BEV`: [scenario_suite_small_global_image_only.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/scenario_suite_small_global_image_only.json)
- `state + ego`: [scenario_suite_small_ego_fp_home_0_state_text.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/scenario_suite_small_ego_fp_home_0_state_text.json)
- `no_state + ego`: [scenario_suite_small_ego_fp_home_0_image_only.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/scenario_suite_small_ego_fp_home_0_image_only.json)

## Ablation 007

### Question

如果复刻原来的 oracle correctness 状态集合和状态文本，只把图像输入从 `HyperGym BEV map` 换成 `IsaacGym top-down RGB`，VLM 输出会不会发生明显变化？

### Compared Conditions

- `hyper_bev + image_with_state_text`
- `isaac_topdown + image_with_state_text`

### Shared Evaluation Setup

- 固定轨迹 seed：`[7, 8, 9]`
- `episodes = 3`
- `max_steps = 30`
- `samples_per_episode = 6`
- `num_home = 2`
- `num_away = 2`
- 总样本数：`18`
- 总球员动作预测数：`36`
- 状态文本：和原 oracle correctness 实验相同
- 唯一变化：图像来源

### Results

| Condition | policy_accuracy | step_exact_match_rate | target_hit_rate | mean_target_error |
| --- | ---: | ---: | ---: | ---: |
| `hyper_bev + image_with_state_text` | `0.8611` | `0.7222` | `0.7222` | `0.6221` |
| `isaac_topdown + image_with_state_text` | `0.9167` | `0.8333` | `0.6111` | `0.8701` |

### Direct Output Difference

不只是相对 oracle 的指标不同，VLM 输出本身也明显变了：

- `16 / 18` 个样本至少有一个球员预测发生变化
- `19 / 36` 个球员预测发生变化

典型变化包括：

- 原来同样是 `move_to_target`，但 target 点改了
- 一些 `move_to_target` 和 `trap_ball` 之间发生了切换

### Interpretation

这组结果说明：

- 把图像源从 `HyperGym BEV` 换成 `IsaacGym RGB`，确实会显著改变 VLM 输出
- 而且这种变化不是小抖动，因为它影响到了过半样本
- 在这批样本上，`Isaac top-down` 反而提高了动作类别匹配率
- 但目标点更偏，说明它更容易选对“大类动作”，不一定更接近 oracle 的具体落点

更谨慎的说法是：

- 图像域变化已经成为有效变量
- 当前 VLM 对图像表现形式并不鲁棒
- 同一底层局面，换一种视觉渲染风格后，它的高层策略会发生可观变化

这里需要强调一个解释边界：

- `target` 并不是唯一标准答案
- scripted oracle 给出的 target 只是“其中一种可行战术落点”，不是唯一正确解

因此在这组对照里，`policy_accuracy` 和 `step_exact_match_rate` 的解释权重应当高于 target 误差。换句话说：

- 如果 `Isaac top-down` 让 VLM 更频繁地选中了和 oracle 相同的策略类别
- 即使 target 点和 oracle 的几何误差更大

这仍然说明“把图像输入从 HyperGym BEV map 换成 IsaacGym RGB”是有道理的，而且很可能更接近你真正想测试的视觉输入形态。

### Artifacts

- HyperGym 结果 JSON: [oracle_image_source_hyper_bev_state_text_seed789.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_image_source_hyper_bev_state_text_seed789.json)
- IsaacGym 结果 JSON: [oracle_image_source_isaac_topdown_state_text_seed789.json](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_image_source_isaac_topdown_state_text_seed789.json)
- HyperGym artifact 目录: [oracle_image_source_hyper_bev_state_text_seed789](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_image_source_hyper_bev_state_text_seed789)
- IsaacGym artifact 目录: [oracle_image_source_isaac_topdown_state_text_seed789](/home/ubuntu/jrWork/booster_gym/logs/vlm_poc/oracle_image_source_isaac_topdown_state_text_seed789)
