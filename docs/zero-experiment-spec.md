# 第零实验 Spec：2-Agent 熔断器

- 目标：用最小成本检验 H1 —— 在**总预算严格匹配**下，带"独立复现晋升"的共享知识库是否降低学习成本 L。
- 熔断语义：若本实验看不到 L 的显著下降，**不得**进入 V0.1 的规模化（100-Agent）阶段，先修正理论。
- 日期：2026-10-16。状态：spec 完成，待实现。

---

## 1. 实验设计

### 1.1 分组（配对设计）

- **Treatment**：2 个智能体 + 1 个共享 KB（含复现晋升规则）。
- **Control**：2 个孤立智能体（`kb=None`），其余代码路径完全一致。
- 配对：同一 seed → 同一目标函数（env 的目标抽样只依赖 `seed_target`，两组相同）；智能体内部随机流为 `hash(seed, group, agent_id)`，组间独立。
- 种子数：n = 20 起；若效应方向一致但 p 值边缘，允许按预注册规则追加到 40（见 §6）。

### 1.2 任务族（固定为一种，保证最小性）

- `family = "trig"`：f(x) = Σ 系数·基函数，基函数从固定文法抽取（如 `sin(2x₁)`, `cos(x₂)`, `x₁x₂`, `x₁²` …），项数 3–5，系数 ∈ [-2, 2]。
- 定义域 x ∈ [-2, 2]²；观测 y = f(x) + ε，ε ~ N(0, 0.01²)。
- 智能体的特征库**完备但大**（~30 候选特征，含真项与干扰项）——样本效率成为关键，共享"已验证项"才有价值。
- 保留测试：21×21 固定网格（reset 时确定）；θ* = 0.01（MSE）。

### 1.3 预算（硬约束，代码中断言）

- 每智能体查询预算 B_q（经 §5 校准，初值 80）；查询成本 c_q = 1。
- **复现查询计入复现者自己的预算**——验证是有价格的（H2 的成本侧）。
- KB 发布/拉取操作记通信计数，不计入查询预算，但如实报告（为 V0.1 的 C(t) 做准备）。
- 两组**总查询预算恒等**：`assert total_queries(treatment) == total_queries(control)`。

### 1.4 运行循环（runner，每轮）

```
for round in range(max_rounds):
    for agent in agents:
        agent.step()                       # 选点→查询→观测→重拟合信念
        for rec in agent.propose():        # M → K：输出候选记录
            kb.publish(rec)                # kb=None 时为 no-op
    for agent in agents:                   # 治疗组：拉取已验证记录
        for rec in kb.pull(agent.id, k=5): # 仅"已复现"、非自己提出的
            agent.adopt(rec)
    for agent in agents:                   # 复现（治疗组）
        for cand in kb.candidates_from_others(agent.id):
            res = agent.replicate(cand)    # 用全新查询，花自己预算
            kb.report_replication(cand.id, agent.id, res)
    if all(a.mse_test <= theta_star): break
```

---

## 2. 接口定义

### 2.1 环境 `environment/lawlab.py`

```python
@dataclass(frozen=True)
class EnvConfig:
    family: str = "trig"
    domain: tuple = ((-2.0, 2.0), (-2.0, 2.0))
    noise_std: float = 0.01
    query_cost: float = 1.0
    query_budget: int = 80          # §5 校准后覆盖
    test_grid_n: int = 21
    seed_target: int = 0            # 只决定目标函数；两组相同 seed 下必须一致

class BudgetExhausted(Exception): ...

class LawLabEnv:
    def __init__(self, config: EnvConfig) -> None: ...
    def reset(self) -> None: ...                       # 抽取目标 f，清零计数器
    def query(self, x: np.ndarray) -> float: ...        # x shape (2,); 超预算抛 BudgetExhausted
    def evaluate(self, predict_fn: Callable[[np.ndarray], float]) -> float: ...
        # 在保留测试网格上的 MSE；predict_fn 接受 shape (2,) 返回 float
    @property
    def queries_used(self) -> int: ...
    @property
    def budget_remaining(self) -> int: ...
    # 注意：target 函数本身不对 Agent 暴露；仅 oracle/分析可用 env._target（下划线，
    #       测试与 verdict 代码可用，Agent 代码禁止引用——用 lint 或代码审查保证）
```

### 2.2 智能体 `agents/agent.py`

```python
@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    feature_library: list[str]   # 如 ["1","x1","x2","x1^2","x2^2","x1*x2","sin(2*x1)",...] (~30)
    solver: str = "omp"          # 正交匹配追踪；备选 "lasso_cv"（消融用）
    committee_size: int = 5      # query-by-committee 的成员数（bootstrap/不同稀疏度）
    candidate_pool_n: int = 200  # 每轮从中选查询点的候选池
    seed_agent: int = 0

@dataclass
class BeliefState:
    committee: list[dict]        # 每个成员：{support: 特征子集, coef: 向量, train_mse}
    memory_n: int                # 情景记忆条数
    protected_terms: set[str]    # 已采纳的验证项（拟合时强制保留）
    mse_test: float              # 上次 evaluate 结果（runner 回填，仅日志用）

class Agent:
    def __init__(self, cfg: AgentConfig, env: LawLabEnv, kb: "KnowledgeBase | None") -> None: ...
    def step(self) -> dict: ...
        # 1) committee 在候选池上投票，选分歧最大点 x（分歧=成员预测方差）
        # 2) y = env.query(x)；追加情景记忆；3) 重拟合 committee（含 protected_terms）
        # 返回 StepLog dict（x, y, committee分歧, 耗时…），供 JSONL 日志
    def propose(self) -> list["KnowledgeRecord"]: ...
        # M→K：从 committee 共识中提取候选记录：
        #   - kind="term"：被 ≥80% 成员选中的项，附系数均值±std
        #   - kind="expression"：committee 最佳成员的完整表达式（MSE 最低者）
        # 去重：与已发布/已验证预测等价的不重复发布
    def replicate(self, record: "KnowledgeRecord") -> "ReplicationResult": ...
        # 用 ≥15 个全新查询（与 record.evidence 的 query_ids 无交集，花自己预算）独立检验：
        #   - term：自己稀疏拟合是否选中该项，且 |coef_rep - coef_prop| ≤ 25% 相对误差
        #   - expression：在新查询上 MSE ≤ theta_rep (=0.05，config 可配，预注册)
        # 返回 ReplicationResult(success, mse_on_fresh, n_fresh_queries, fresh_query_ids)
    def adopt(self, record: "KnowledgeRecord") -> None: ...
        # 采纳已验证记录：term → 加入 protected_terms（强制保留，系数用自己数据重估）；
        # expression → 加入 committee 候选池并优先保留
    @property
    def belief(self) -> BeliefState: ...
    def predict(self, x: np.ndarray) -> float: ...   # committee 均值，供 env.evaluate
```

### 2.3 知识记录与 KB `culture/kb.py`

```python
@dataclass
class Provenance:
    proposer_id: str
    generation: int          # 第零实验恒为 0，为 V0.1 预留
    parent_ids: list[str]
    seed: int

@dataclass
class ReplicationResult:
    record_id: str
    replicator_id: str
    success: bool
    mse_on_fresh: float
    n_fresh_queries: int
    fresh_query_ids: list[str]

@dataclass
class KnowledgeRecord:
    id: str
    kind: str                # "term" | "expression"
    proposition: str          # 人类可读，如 "term sin(2*x1): coef=1.30±0.12"
    expression: Callable | None  # 可执行：term 求值器 / 完整预测函数
    evidence: dict           # {n_queries, residual_mse, query_ids, domain}
    conditions: str          # 适用条件，如 "x in [-2,2]^2, trig family"
    provenance: Provenance
    confidence: float
    status: str              # "candidate" | "validated" | "refuted" | "superseded"
    replications: list[ReplicationResult]
    counterexamples: list
    open_questions: list[str]

class KnowledgeBase:
    def publish(self, record: KnowledgeRecord) -> str: ...
        # 分配 id；预测等价去重（与现有记录在探测集上预测一致则合并证据，不新增）；
        # status="candidate"
    def candidates_from_others(self, agent_id: str) -> list[KnowledgeRecord]: ...
        # status=candidate 且 proposer != agent_id
    def report_replication(self, record_id: str, agent_id: str, res: ReplicationResult) -> None: ...
        # 晋升规则（预注册，不可调）：
        #   - 来自 proposer 之外的 ≥1 次 success → "validated"
        #   - 累计 ≥2 次 fail → "refuted"
        #   - 断言：replicator != proposer（自己不能复现自己）
    def pull(self, agent_id: str, k: int) -> list[KnowledgeRecord]: ...
        # 仅 status="validated" 且 proposer != agent_id，按 confidence 排序取 k 条
    def get_validated(self) -> list[KnowledgeRecord]: ...
    @property
    def replication_rate(self) -> float: ...   # 成功复现次数 / 总复现次数（诊断用）
```

### 2.4 指标 `evaluation/metrics_zero.py`（纯函数）

```python
def learning_cost(query_idx_first_below_theta: int | None, c_q: float) -> float: ...
    # 首次 MSE ≤ θ* 时的累计查询成本；从未达到 → inf（统计时按预算上限+惩罚处理，预注册）
def median_ci(values: list[float], n_boot: int = 10000) -> tuple[float, float, float]: ...
def paired_permutation_test(diffs: list[float], n_perm: int = 20000, alternative="greater") -> float: ...
    # H0: median(d) ≤ 0，其中 d_i = L_control_i − L_treatment_i（配对差）
def disagreement(a: Agent, b: Agent, probe_X: np.ndarray) -> float: ...
    # 探测集上预测差的均方根（D 的第零实验版本）
```

### 2.5 实验脚本 `experiments/run_zero.py` + `configs/zero.yaml`

```yaml
# zero.yaml（预注册文件：开跑前冻结，git tag）
n_seeds: 20
theta_star: 0.01
theta_rep: 0.05
query_budget_per_agent: 80   # §5 校准后填写
max_rounds: 40
pull_k: 5
min_fresh_queries_for_replication: 15
decision_rule:  # §6，逐字执行
  alpha: 0.05
  disagreement_floor_ratio: 0.5
```

- 日志：`runs/<timestamp>-<gitsha>/seed_{s}/{treatment,control}.jsonl`，append-only，每行含 schema_version。
- 断点续跑：已完成的 seed 跳过；重跑已完成 seed 结果 bit-identical（除 wall-clock）。

---

## 3. 实现顺序（对应工程计划第 1–4 步的最小子集）

1. `environment/lawlab.py` + `tests/test_env.py`
2. `agents/agent.py`（solver=omp, committee）+ `tests/test_agent.py`
3. `culture/kb.py`（最小 KB：schema + 晋升状态机）+ `tests/test_kb.py`
4. `experiments/run_zero.py` + `evaluation/metrics_zero.py` + `tests/test_harness.py`
5. 校准（§5）→ 冻结 `zero.yaml` → 正式跑 → `verdict.py` 输出判决书

---

## 4. 测试清单

### 环境
- [ ] 相同 `seed_target` → 相同目标函数（比较抽样系数向量精确相等）。
- [ ] 相同 seed 下查询序列的观测值可复现（含噪声）。
- [ ] 超预算查询抛 `BudgetExhausted`，计数器不脏。
- [ ] `evaluate` 使用的测试网格在 reset 时固定、两组一致。

### 智能体
- [ ] 孤立智能体在 trig 族上学习曲线大致单调，能在预算内达到 θ*（校准依据）。
- [ ] 稀疏拟合在真函数数据上恢复支持集（precision/recall ≥ 0.9，稠密网格 oracle 测试）。
- [ ] `propose()` 输出的记录可执行：`record.expression` 在测试网格上可算出 MSE。
- [ ] `replicate(真记录)` → success；`replicate(伪造记录)` → fail（构造错误项/错误系数）。
- [ ] `adopt()` 后 protected term 在后续拟合中被强制保留（即使自己数据少）。

### KB
- [ ] 伪造记录永为 candidate（无人复现成功则永不晋升）。
- [ ] 一次他者成功复现 → validated；两次失败 → refuted。
- [ `report_replication` 中 replicator == proposer 时抛断言错误。
- [ ] `pull()` 永不返回 candidate/refuted/自己提出的记录。
- [ ] 预测等价的两条记录 publish 后合并为一条（证据合并，id 不增）。
- [ ] provenance 字段完整（proposer/generation/parent_ids/seed 缺一不可）。

### Harness 与统计
- [ ] 配对 seed 下两组目标函数系数向量精确相等（防配对失效）。
- [ ] 两组总查询预算精确相等（`assert`，不是约等）。
- [ ] 日志 schema_version 一致；杀掉重跑 bit-identical。
- [ ] 合成数据测试：注入已知效应（treatment L = 0.7×control）→ verdict=GO；零效应 → verdict=STOP；零效应重复 200 次 → 一类错误率 ≈ α（±容差）。
- [ ] L=inf（从未达标）种子的处理与预注册一致（按 `query_budget*c_q + 惩罚项` 计，惩罚项写死）。

---

## 5. 校准程序（正式跑之前，必须做）

1. 孤立智能体 × 10 seeds，记录达到 θ* 所需查询数。
2. `query_budget_per_agent` = 第 90 百分位数（~90% 孤立运行可达标——避开地板/天花板效应）。
3. Oracle 检查：用稠密网格拟合，真模型 MSE < θ*/10——否则是求解器/特征库 bug，不是社会机制问题，先修。
4. 把校准数值填入 `zero.yaml` 并冻结（git tag `zero-preregistered`）。

## 6. 预注册判决规则（`verdict.py` 逐字执行）

输入：20 个配对差 d_i = L_control_i − L_treatment_i。

- **GO**：单侧配对置换检验 p < 0.05（treatment 降低 L），**且** treatment 终期 disagreement ≥ 0.5 × control 终期 disagreement（未完全坍缩）。→ 进入 V0.1。
- **CAUTION**：p < 0.05 但 disagreement 跌破下限。→ 共享有效但同质化；V0.1 规模化前必须先加入多样性保持机制。
- **STOP**：p ≥ 0.05。→ H1 在此设定下不成立；**不得规模化**；先看诊断再修理论。
- 边缘追加规则：若 0.05 ≤ p < 0.15 且中位差方向为正，允许追加 20 seeds（共 40）重判一次；否则不许追加（防 p-hacking）。

### STOP 后的诊断清单（区分"理论错"与"实验没做好"）

1. `replication_rate` 是否过低（<0.5）？→ 晋升规则或 propose 质量问题，不是 H1 的问题。
2. control 组是否大量顶到预算天花板 / 治疗组是否已贴地板？→ 任务难度/预算校准问题，重校准再跑。
3. 两智能体查询点重叠度是否 ≈1？→ 探索策略本身无多样性，共享无信息增量——这是真阴性，修理论。
4. 逐 seed 检查：treatment 在哪些 seed 上赢/输？失败 seed 的目标函数有何特征（如某类项难发现）？

---

## 7. 完成标准

- [ ] §4 测试清单全部通过。
- [ ] §5 校准完成，`zero.yaml` 冻结打 tag。
- [ ] 20 seeds 跑完，`verdict.py` 输出 GO / CAUTION / STOP 判决书（含 p 值、CI、诊断表）。
- [ ] 判决书与原始日志一并存档到 `~/workspace/goals/ai/files/omc/zero-run-<date>/`。
- 任何情况下**不以"代码跑通"为完成**，以判决书为准。
