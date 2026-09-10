# INTERVIEW_NOTES · 求职与面试应对手册

面向「GenAI / RAG / Agent 工程师」岗位的简历话术、技术决策与高频追问标准答案。把本项目当作作品讲解时用。

---

## 一、简历 Bullet 示例（可直接用）

- 设计并实现 **LangGraph 多 Agent 职业助手**：以 `StateGraph` 完成「分类 → 路由 → 叶子节点」工作流，覆盖教程 / 答疑 / 简历 / 面试题 / 模拟面试 / 职位搜索 6 条业务线，并新增 `fallback` 兜底节点处理未识别输入。
- 从零搭建 **企业知识库 RAG 引擎**：完成文档解析 → 清洗 → 切分 → Embedding → FAISS 建库全链路，落地 **BM25 + 向量混合检索 + RRF 融合 + LLM/API 双 Rerank**，回答带文件级引用溯源，支持多知识库隔离与增量入库。
- 工程化去阻塞改造：将 Notebook 内 `while True + input()` 阻塞循环重构为 `SessionManager.start/step/finish` 单步状态机，使同一套业务逻辑在 **Streamlit 与 FastAPI 两端复用**，无两套实现漂移。
- 建立 **可复现评测体系**：基于知识库 chunk 反向生成评测集（LLM 生成 + 人工校对），固定 `Recall@5 / MRR / Hit Rate / 幻觉率（LLM-as-judge）` 口径，输出「向量 vs 混合检索」「无 / LLM / API Rerank」A-B 对比报告。
- 稳定性与成本治理：落 `tenacity` 重试退避、信号量限流、超时控制、`diskcache` 结果缓存、`tiktoken` token 成本统计、内容安全过滤与脱敏日志（不记录 Key / 文档全文）。

---

## 二、技术决策话术（为什么这么选）

### Q：为什么用混合检索（BM25 + 向量）而不是纯向量？
> 向量检索擅长语义泛化，但对**专有名词、岗位名、技术栈精确匹配**容易漏召回。BM25 是词面精确匹配，互补性强。两者用 **RRF（Reciprocal Rank Fusion）** 融合——只用排名、不调权重，避免向量分数与 BM25 分数量纲不一带来的权重调参难题，也更稳健。

### Q：RRF 公式是什么？
> `RRF(score, d) = Σ_{r∈R} 1 / (k + rank_r(d))`，k 通常取 60。对每个召回列表按文档排名取倒数求和，排名越靠前贡献越大。不依赖分数绝对值，只依赖相对排名。

### Q：为什么要做 Rerank？为什么用 LLM Rerank 而不是 CrossEncoder？
> 融合后 top-K 仍可能混入弱相关片段。CrossEncoder 重排效果最好，但需 GPU、本机无独显且内存有限，CPU 推理慢。因此默认用 **LLMReranker**：让对话模型对候选打 0–10 相关性分再排序，零额外依赖、效果可见；同时保留 `APIReranker`（硅基流动 `bge-reranker-v2-m3`）作为可选更强方案。三者本身构成 A-B 实验素材。

### Q：chunk_size / overlap 怎么定？为什么做标题感知切分？
> 默认 `chunk_size=500 / overlap=80`（字符级）。太小丢失上下文、太大稀释相关性且浪费 token。标题感知切分（`RecursiveCharacterTextSplitter` + 标题边界优先）能避免把一个知识点的上下文从中间切断，提升片段内聚性，召回更准。

### Q：Function Calling 与显式检索怎么取舍？
> 两条路径都落地了：默认走**显式检索**（稳定、低延迟、可控可评测）；同时真实实现 `bind_tools()` + `AgentExecutor` 版本供需要自主规划的场景。显式检索更适合可度量、可审计的 RAG 链路；Function Calling 更适合开放任务。取舍依据是「可控性 vs 自主性」。

### Q：为什么 Embedding 走云端而不是本地？
> 本机无独显、内存 8–16G，本地 `bge-m3` 需 torch（1–2GB）且 CPU 推理慢，新手劝退。改为云端 `/embeddings`（硅基流动 `BAAI/bge-m3`），`faiss-cpu` 做向量库，10 分钟装完。用 `EmbeddingFactory` 抽象保留本地开关，可无缝切换。

---

## 三、高频追问标准答案

### Q：召回率多少？怎么优化的？
> 口径固定：评测集由知识库 chunk 反推（golden = 生成该问题的源 chunk），`Recall@5` = golden 是否在前 5。**典型结果量级**：纯向量约 0.6–0.7，混合检索（RRF）提升 8–15 个百分点；加 LLM Rerank 后 `MRR` 再提升约 5–10%（代价是单轮延迟增加数十到上百毫秒）。具体数字以 `Agent_output/Eval_Report_*.md` 为准——**一定强调口径与评测集来源**，这比数字本身更可信。

### Q：怎么降低幻觉？
> 三道防线：① 检索阶段只用被 Rerank 过的 top-K，减少噪声；② 生成 Prompt 强制「仅依据下列片段作答，不知就说不知道」并附引用；③ 评测阶段用 LLM-as-judge 把回答拆断言、逐条判定是否由片段支撑，统计幻觉率。

### Q：长会话怎么控制成本与延迟？
> `trim_messages(strategy="last", max_tokens=10, token_counter=len, start_on="human", include_system=True)`，只保留最近若干轮，prompt 长度恒定；同一 `(query, 知识库版本, 检索配置)` 走 `diskcache` 命中即返回；Embedding 批量 32–64 调用并缓存。

### Q：多轮对话在 Web / API 怎么驱动？
> 不持有循环。Agent 只暴露 `start(query)` 与 `step(user_text)`；循环控制权交给 UI/API，「用户发一条 = 推进一步」。SessionManager 持有历史与产物，结束时 `finish()` 把对话写入 `Agent_output/{类型}_{时间戳}.md`。

### Q：为什么不用一个超大 Prompt 把所有功能塞进去？
> 职责分离：路由图只分类、Agent 只管单轮生成、RAG 引擎与业务无关可独立评测、API 只做协议转换。这样每层的评测、降级、替换都独立，符合工程中「可度量、可维护」的目标。

---

## 四、现场演示 checklist

1. `streamlit run app.py` 能起、品牌区与状态灯正常。
2. 点快捷示例芯片，确认自动路由命中并出现路由提示条。
3. 上传一份示例文档 → 建库 → 提问，确认出现「引用来源」区块与原文摘录。
4. 侧边栏切「重排方式 / top_k」→ 重新提问，确认检索可视化变化。
5. 点「运行评测」→ 右侧出现 `Eval_Report_*.md`，展示 A-B 对比表。
6. `/docs` 能打开，`/health` 返回模型连通状态。
