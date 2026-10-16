# 第零实验判决书（熔断器）

- 日期：2026-10-16
- 配置：`omc-zero/configs/zero.yaml`（已冻结于 `runs/formal20/zero.yaml.frozen`）
- 原始日志：`omc-zero/runs/formal20/seed_*/{treatment,control}.jsonl`
- 判决：**CAUTION**（按预注册规则逐字执行）

## 判决数据

| 指标 | 值 |
|---|---|
| 配对置换检验 p（单侧） | 0.00005 |
| L 差中位数（control − treatment） | 8.25 queries |
| 95% CI | [1.75, 19.25]（不含 0） |
| 胜/负/平（20 seeds） | 17 / 1 / 2 |
| disagreement 比（treat/ctrl） | 0.386（< 0.5 → 触发 CAUTION） |
| 治疗组平均复现率 | 0.786 |
| control 撞预算天花板 | 12/20 seeds |

## 解读

**H1 在此设定下成立**：带"独立复现晋升"的共享知识库显著降低学习成本（p=5e-05）。

**CAUTION 的来源不是有害坍缩**：disagreement 比 <0.5 是因为——
- treatment 成功的 18 个 seed 上，两个智能体都收敛到**真函数**，分歧自然低（均值 0.075）；
- treatment 失败的 2 个 seed 上，分歧依然高（0.39/0.76），**没有锁死在同一个错误信念上**；
- control 的高分歧来自 12/20 seeds 撞天花板后"错得各不相同"，不是健康多样性。

即：此处的"分歧下降" = 收敛到真值，而非 groupthink。但预注册规则按字面执行为 CAUTION，
其行动含义正确——**进入 V0.1 时必须内置多样性保持机制**（更难、更开放的任务中，
过早收敛是真风险）。

## 生效的机制（事后分析）

1. **互补分享**，而非"领导-跟随"：term 级记录（委员会全票通过的项）在早期分享，
   A 分享 2 个项、B 分享另 2 个，合并后双方解出 control 双双失败的任务
   （seeds 2, 3, 7, 8, 16, 19 均为此模式）。
2. **验证有效**：BIC 独立验证的真阳性率 100%（原型测试），正式跑中复现率 0.79，
   未观察到错误知识污染。
3. **预算严格匹配**：两组每智能体 task 查询 cap 完全相同（25），验证为免费计算
   （用 replicator 自己的独立数据），无预算作弊。

## 校准中发现并修复的真问题（已记入测试）

1. OMP 在相关字典上贪心选错 → 改为 OMP + BIC 后向剪枝（`test_omp_bic_recovers_exact_support_dense`）。
2. 预算匹配 bug：`total_cap` 曾按组计算导致两组总额不等 → 加断言与专项测试。
3. term 复现在 fresh-query 下不可靠 → 改为 replicator 自有数据的 BIC 针对性检验；
   expression 复现为纯预测检验（6/6 真阳性，0/6 假阳性）。
4. 候选泛滥（32 候选/1 验证）→ propose 门控（train_mse + 实质突破）+ 每轮只复现最高置信候选。

## 下一步（按 CAUTION 执行）

- 熔断器**通过**：允许进入 OMC-V0.1，但 V0.1 设计必须包含多样性保持机制
  （ collapse 与 polarization 双指标监控，见研究交接文档 §4.5）。
- V0.1 优先级：LawLab 多任务族 + Null-Civ 对照 + 跨文明信息流（H3），
  复用本实验的 KB/验证/预算匹配基础设施。
