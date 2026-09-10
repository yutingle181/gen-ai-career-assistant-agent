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

---

## 五、系统整合与故障排查（软硬一体视角）

> 当被问到「你的项目整体怎么搭、前后端和 Agent 怎么连、本机踩了什么坑」时，用这组。前四节偏 Agent/RAG 内部，本节偏「软件 + Agent 协同」与运维实操。

### Q：为什么 Agent 不嵌进 Java 后端，而是独立 Python 服务 + 薄网关？
> 技术栈与迭代节奏不同：Agent 是 LangGraph/FastAPI/RAG，后端是 Spring Boot 3.2.5 + Vue3，各自演进互不污染。Java 只做它擅长的「存、管、看」（账号/JD/简历/归档/站点），Agent 的 7 个能力一律不重写。代价是多一个服务编排（先起 :8000 再起 :8080），换来两侧可独立部署、测试。

### Q：薄网关除了转发还做了什么？为什么只做这一件？
> 唯一加工点是「装配用户画像」：`AgentContextService` 按登录用户把「所选岗位 JD + 关联简历正文」拼成 `user_context`，注入每轮 `/chat`、`/chat/stream`。其余零加工、Agent 代码零改动。这样 Agent 才能结合用户背景作答，而 Java 不背负任何 GenAI 逻辑，避免能力重复实现、职责纠缠。

### Q：/api/job-sites 之前为什么 30s 超时？怎么定位修的？
> 前端报 `timeout of 30000ms exceeded`，但接口逻辑很轻。`jstack` 抓到请求线程卡在**日志写出**——`com.jobseeker` 开 DEBUG 后 MyBatis 每条 SQL 打日志，控制台 appender 同步阻塞，高并发下线程被锁。修法：① `logback-spring.xml` 改 `AsyncAppender`（`neverBlock=true`），日志写出不阻塞业务线程；② `application.yml` 把 `com.jobseeker` 从 DEBUG 降 INFO。前端再加 8s 请求级超时 + 加载/失败重试三态，避免把「超时」误报成「无站点」。

### Q：用户「岗位/简历」背景怎么进 Agent？历史长了画像丢不丢？
> 前端在「AI 助手」页选岗位 + 关联简历 → 网关校验归属后取 JD/简历截断拼 `user_context` → 随请求发送。Agent 侧 `SessionManager` 每轮把 `user_context` 前置为 `SystemMessage`，且**历史裁剪后画像不丢**（画像独立于聊天记录、每轮重注入）。`user_context` 为空时行为与旧版一致。

### Q：流式输出怎么落地？为什么不在容器外先渲染？
> 链路：模型 `llm_stream` → `SessionManager.start_stream/step_stream` → SSE `/chat/stream`。关键：提交输入只「入队」，真正生成放在对话容器内部执行，token 直接落在助手气泡里，不会先渲染到容器外、结束再跳进消息列表。流式中途失败给友好提示而非抛栈；完整回复流结束才入栈，避免半截回答进下一轮上下文。

### Q：Agent 接口为什么没有 /api 前缀？调用方注意什么？
> 真实路径是 `/chat`、`/knowledge`、`/sessions`（旧文档写 `/api/chat` 是错的，已修正）。网关注发就是无前缀路径；`chat/knowledge/sessions` 三组带 `X-API-Key` 鉴权（`API_KEY` 空时关闭）。Agent 必须先于后端启动（:8000），否则网关注入 `user_context` 时连不上。

### Q：本机 D: 盘为什么导致 Vite 504、Maven 编译失败？怎么绕？
> 本机安全过滤驱动**全局禁止 D: 盘文件重命名（MoveFile）**。Vite 预构建把 `deps_temp_*` 重命名为 `deps` 被拦 → 持续 504；Maven `resources-plugin` 用「临时文件 + rename」原子拷贝被拦 → `AccessDeniedException`，且 Safe-Delete 拒绝删 `target`。绕法：Vite `cacheDir` 指向 C: 盘；Maven 用 `-Dmaven.resources.skip=true` 或导 classpath 走 `java -cp` 启动；构建产物 `<build><directory>C:/jobseeker-target</directory>` 改到 C: 盘。
