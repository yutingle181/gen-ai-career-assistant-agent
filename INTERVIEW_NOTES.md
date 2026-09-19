# INTERVIEW_NOTES · 求职与面试应对手册

面向「GenAI / RAG / Agent 工程师」岗位的简历话术、技术决策与高频追问标准答案。把本项目当作作品讲解时用。

---

## 一、简历 Bullet 示例（可直接用）

- 设计并实现 **LangGraph 多 Agent 职业助手**：以 `StateGraph` 完成「分类 → 路由 → 叶子节点」工作流，覆盖教程 / 答疑 / 简历 / 面试题 / 模拟面试 / 职位搜索 6 条业务线，并新增 `fallback` 兜底节点处理未识别输入。
- 从零搭建 **企业知识库 RAG 引擎**：完成文档解析 → 清洗 → 切分 → Embedding → FAISS 建库全链路，落地 **BM25 + 向量混合检索 + RRF 融合 + LLM/API 双 Rerank**，回答带文件级引用溯源，支持多知识库隔离与增量入库。
- 工程化去阻塞改造：将 Notebook 内 `while True + input()` 阻塞循环重构为 `SessionManager.start/step/finish` 单步状态机，使同一套业务逻辑在 **Streamlit 与 FastAPI 两端复用**，无两套实现漂移。
- 建立 **可复现评测体系**：基于知识库 chunk 反向生成评测集（LLM 生成 + 人工校对），固定 `Recall@5 / MRR / Hit Rate / 幻觉率（LLM-as-judge）` 口径，输出「向量 vs 混合检索」「无 / LLM / API Rerank」A-B 对比报告。
- 稳定性与成本治理：落 `tenacity` 重试退避、信号量限流、超时控制、`diskcache` 结果缓存、`tiktoken` token 成本统计、内容安全过滤与脱敏日志（不记录 Key / 文档全文）。
- 打通 **Function Calling 自主工具调用**（`bind_tools` + LangGraph 条件边，轮次上限 / 工具异常回灌 / 超时降级三重边界），以 `ENABLE_TOOL_CALLING` 开关与显式检索**双路径共存**，并用 15 条样本实测出 **延迟 -56%、token 0.22x** 的量化结论；工具调用过程经 SSE `tool_call` 事件在两个前端渲染为可折叠时间线。
- 新增 **JD 匹配诊断**与**面试复盘**两个结构化场景（Pydantic `with_structured_output` + 前端评分卡 / 四段式复盘卡片），并通过 Spring Boot 薄网关把 MySQL 中的岗位 JD 与简历装配成 `user_context` 注入提示词，实现 **AI 编排与业务数据的安全联动**；联调中定位并修复雪花 ID 精度丢失、流式回复未入栈/未落库、长任务被网关读超时掐断等 4 个跨端缺陷。

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
> **两条路径都在跑，且能切、能观测、能量化。** 默认走**显式检索**（先检索一次再生成，稳定、可审计、可评测）；打开 `ENABLE_TOOL_CALLING=true` 后改由模型通过 `bind_tools()` **自主决定**是否检索、检索什么关键词、检索几轮。
>
> 循环用 **LangGraph `StateGraph` 条件边**（`assistant → tools → assistant`）实现，而不是 `AgentExecutor` 或手写 `while`——项目约定「多轮人机推进权归 `SessionManager`」，工具调用是**单轮内的推理闭环**，用图表达既满足 ReAct 语义又不动既有约定，也少一层与项目风格不符的抽象。三条安全边界：轮次上限 3（超限强制收束为直接作答）、单步工具沿用守护线程超时、工具异常只回灌一句中文提示（堆栈留在日志，避免异常细节进上下文推高 token）。
>
> **实测（15 条样本，两条路径钉同一模型 `qwen-turbo`，重排关闭以排除干扰）**：显式检索 6765 ms / 2392 token；Function Calling **2992 ms / 528 token**——延迟 **-56%**，token **约 0.22x**。原因不是「FC 模型更聪明」，而是显式路径**每次都**把 top-K 片段整段塞进 prompt，而 FC 只在模型判断确有必要时才付这次开销（实测仅 **26.7%** 的样本触发检索，平均 1.27 轮）。这条数字与口径提醒都写进了 `Agent_output/Tool_AB_Report_*.md`。
>
> 取舍结论：**成本敏感 + 确定性强的知识库问答留显式检索；开放式、需要多源信息的问题交给 Function Calling**。所以开关默认关闭——不赌模型每次都判断对，而是让两种策略可回滚、可对比。
>
> （顺带一个真话：改造前 `get_agent_tools()` 全仓库**没有任何调用点**，即"写了 Function Calling 却根本没接上"，而且它塞进去的 `DuckDuckGoSearchResults` 还缺 `web_search()` 那套超时保护——同一件事两条路径行为不一致。这次先把工具层收敛成单一安全实现，再谈接链路。）

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

### Q：AI 怎么跟业务数据联动？（JD 匹配诊断）
> 业务数据在 MySQL（岗位 JD + 简历），AI 编排在 Python，中间隔着 Spring Boot 薄网关。网关只做一件事：把前端传的 `position_id / resume_id` **换成 Agent 认识的 `user_context`** 文本（查库 + 拼装，且校验归属当前用户防越权），提示词如何消费由 Agent 决定——网关里没有一行 AI 逻辑。
>
> Agent 侧用 Pydantic `with_structured_output` 产出结构化结果（总分 / 分项 / 命中项 / 缺口项 / 面试准备重点），**同时**渲染成 Markdown 供对话展示与落盘；结构化产物随 SSE `done` 事件下发 `structured` 字段，前端据此渲染**评分卡卡片**（环形总分 + 分项进度条 + 命中/缺口双栏），而不是让用户去读一大段文字。实测：关联岗位 + 简历后返回 73/100（技能 80 / 经验 70 / 学历 0 / 项目 80）。
>
> 一个细节值得讲：结构化输出**失败时不能空手而归**——`with_structured_output` 异常或返回类型不对时自动回退为纯文本，且提示词里写死了 Markdown 小标题契约，因此**回退路径下前端仍能解析出同样的分区卡片**（复盘页就是这么做的：优先解析结构化 Markdown，失败则回退到后端抽取的「建议 / 弱点」字段）。

### Q：面试复盘为什么不是「再做一个页面」？
> 因为那会变成第二条重复链路。已有链路是「会话 → 归档 → 复盘字段（suggestions / weaknesses / detail）」，**断点在中间**：归档的只是纯文本，模型从没加工过它；而且流式接口此前根本没把 AI 回复存进会话（见 §五）。所以这次做的是「补断点」：Agent 新增 `interview_review` 模式产出结构化复盘，Java 侧的抽取逻辑改成**按小标题分段**（命中「薄弱点 / 改进动作 / 建议」的标题会把它下面的条目一并带出），前端把同一份文本解析成**四段式卡片**（表现评分环形指标 / 追问链步骤条 / 薄弱点警示列表 / 改进动作可勾选清单）。存量老数据没有小标题，则自动回退到「建议 / 弱点」两块——**不破坏任何历史记录**。

---

## 四、现场演示 checklist

1. `streamlit run app.py` 能起、品牌区与状态灯正常。
2. 点快捷示例芯片，确认自动路由命中并出现路由提示条。
3. 上传一份示例文档 → 建库 → 提问，确认出现「引用来源」区块与原文摘录。
4. 侧边栏切「重排方式 / top_k」→ 重新提问，确认检索可视化变化。
5. 点「运行评测」→ 右侧出现 `Eval_Report_*.md`，展示 A-B 对比表；勾上「同时跑工具调用 A/B 对比」再跑一次，报告里会多出「显式检索 vs Function Calling」章节（延迟 / token / 轮次 / 成功率）。
6. `/docs` 能打开，`/health` 返回模型连通状态。
7. **三个新界面**（jobseeker 侧，需 :8000 → :8080 → :5173 顺序启动）：
   - 关联「岗位 + 简历」后选 `JD 匹配诊断` 提问 → 出现评分卡（环形总分 + 分项进度条 + 命中/缺口双栏 + 面试准备重点）。
   - 选 `面试复盘` 贴一段面试记录 → 点「归档到复盘」→ 到「面试复盘」页展开，看到四段式卡片。
   - `ENABLE_TOOL_CALLING=true` 时选 `职位搜索` 提问 → 消息下方出现折叠时间线「调用了 N 个工具 · Xs」，展开可见工具名/耗时/入参/返回摘要。
   - 截图存于 `docs/screenshots/`（`jd-scorecard.png` / `tool-timeline.png` / `review-cards.png`），演示环境无网时也能用图讲。

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

### Q：联调时遇到过什么「最难查」的 Bug？
> 四个，都是「界面看着正常、数据其实丢了」的类型，讲出来比背概念有说服力：
>
> 1. **雪花 ID 在 JS 里被四舍五入**：主键是 19 位 Long（`...587778`），浏览器 `JSON.parse` 后变成 `...587800`，回传时 `position_id` 已经是错的 → 网关查不到岗位 → `user_context` **静默**注入失败 → 模型回答「你还没关联岗位」。前端看请求体完全正常、日志也不报错，只能靠**对比前后端 ID 的实际取值**才发现。修法：`JacksonConfig` 全局把 Long 序列化为字符串（`int` 不受影响，`Result.code` 仍是数字），前端把 ID 当不透明字符串透传。
> 2. **流式接口绕过了会话层**：`/chat/stream` 直接调 `agent.respond_stream()`，于是生成完的**完整回复既没进 `history` 也没进 `record`**——多轮里模型看不到自己上一轮说了什么，归档接口也拿不到任何 AI 产出（复盘永远为空）。修法：改走 `SessionManager.start_stream/step_stream(auto_finish=False)`，保留「导出由用户显式触发」的既有交互。
> 3. **流式回复没落库**：归档读的是消息表，非流式接口一直在写，流式接口漏了。修法：流结束后补 `db.add_message(sid, "assistant", text)`。
> 4. **长结构化调用被中间层掐断**：结构化输出在首个 token 前可能静默很久，超过网关 OkHttp 的 120s 读超时 → 网关抛 `BizException: Agent 流式调用失败：timeout`，而 SSE 响应头早已发出，异常只能变成「连接挂着不动」。修法：`/chat/stream` 每 15s 发一条 SSE 注释心跳（`: keep-alive`，不产生事件、前端自然忽略）保活，把读超时从「硬上限」变成「静默上限」。
>
> 共同点都是**先怀疑链路，而不是先怀疑模型**：把 SSE 原始事件流、数据库消息表、前后端 ID 取值三处对齐，问题基本就定位了。

---

## 六、企业级验收标准 × 本项目自评（v1.2.0 生产级补强）

> 用法：面试官问「你的 Agent 能不能上生产」时，**先给验收维度，再说自己落在哪一档**——
> 比笼统说「做得挺全」可信得多。判定：✅ 有机制 / 🟡 有雏形缺一环 / ❌ 当时没有。

### 一条闭环（Agent Loop）

| 验收项 | 补强前 | 补强后 | 证据 |
| --- | --- | --- | --- |
| 循环结构 | ✅ | ✅ | `graph/tool_loop.py`：`assistant → tools → assistant` 条件边 |
| 谁维护状态 | ✅ | ✅ | `ToolLoopState(messages, steps)` + `add_messages`；外层 `SessionManager` |
| 最多多少步 | 🟡 只有轮次上限 | ✅ | `TOOL_CALLING_MAX_STEPS` + **图级 `TOOL_RECURSION_LIMIT`** 两层兜底 |
| 到上限怎么收场 | 🟡 强制收束 | ✅ | `finalize` 节点 + 捕获 `GraphRecursionError` 后 `_forced_finalize`，用户永远拿到结论而不是栈 |
| 工具失败后怎么做 | ❌ 统一降级 | ✅ | **失败分级**：超时/网络/5xx 重试 → 换参数（简化检索式）→ 换通道（backend）→ 如实降级 |
| 失败怎么让用户知道 | 🟡 一句提示 | ✅ | 降级文本带 `【工具降级:kind】`，事件流 `ok=False` + `error_kind`，Prompt 要求声明「未联网核实」 |

### 两个外壳（Runtime / Harness）

| 验收项 | 补强前 | 补强后 | 证据 |
| --- | --- | --- | --- |
| 任务生命周期 | 🟡 | 🟡 | `start/step/finish/confirm` + 草稿态；仍无「取消 / 排队」 |
| 限流与并发 | ✅ | ✅ | `Semaphore(4)`（`api/deps.py`） |
| 超时 / 重试 / 退避 | ✅ | ✅ | 守护线程超时 + `tenacity`（4xx 不重试，避免白烧额度） |
| **熔断** | ❌ | ✅ | `tools.CircuitBreaker`：连续失败 3 次冷却 60s，冷却后半开；`/health` 可见 |
| **可恢复性** | ❌ | ✅ | 会话级：`api/deps.restore_session` 从 SQLite 重建；单轮级：`graph/checkpoint.py`（sqlite→memory→不启用自动降级），同轮重试**不重复执行工具** |
| 幂等 | 🟡 | ✅ | `storage.save_file(key=...)`：同会话产物稳定命名，重放只覆盖同一文件（刻意不用 os.replace，本机 D 盘禁 MoveFile） |
| 日志 / 追踪 | ✅ | ✅ | `summarize` 脱敏 + `telemetry.span` |
| **指标** | ❌ | ✅ | `src/metrics.py`：工具失败率 / 熔断拦截率 / 超轮次率 / 递归终止率，`/health` 暴露 |
| 评测 | ✅ | ✅ | `eval/`：Recall@K / MRR / HitRate / 幻觉率 + 双路径 A/B |

### 五套能力

| 能力 | 补强前 | 补强后 | 说明 |
| --- | --- | --- | --- |
| Tool / Function Calling | ✅ | ✅ | `bind_tools` + 双路径 A/B（延迟 −56%、token 0.22x） |
| MCP | ❌ | ❌ | 诚实说：没有。能讲清 FC（调用表达）/ Tool（可执行能力）/ MCP（接入协议）三者边界 |
| Skills | 🟡 | 🟡 | 按 mode 加载人设与工具（按需加载 ✅）；缺注册 / 版本 / 权限骨架 |
| 记忆：短期 | ✅ | ✅ | `history` + `trim_messages` + `user_context` 裁剪后重注入 |
| 记忆：长期 / 槽位 | ❌ | ✅（默认关） | `ENABLE_SLOT_MEMORY`：城市 / 岗位方向 / 时间范围 / 学历单独存并每轮重注入，**不被裁剪丢掉** |
| 多 Agent 准入判断 | ✅ | ✅ | 场景级 agent 工厂 + 中心化路由（串行）；不虚构并行 |
| 并发写仲裁 | ❌ | ❌ | 结构上无并发写（靠串行规避）；能说清「引入并行后：单一写者 + 版本检查 + supervisor 裁决」 |
| **轨迹级评测** | ❌ | ✅ | `eval/trajectory.py`：**TaskSuccessRate**（按场景结构标记 + 工具事件，离线可复现）、**Failure Onset**（最早失败在第几步）、失败原因分布 |

### 讲这套的固定话术

> 「我的项目在**评估、工具调用、可观测、限流、可恢复性**上是生产级，能拿数字自证；
> 在**安全沙箱、并发写仲裁、MCP 接入**上还是 Demo 级，我知道边界在哪、也知道该补什么。
> 比如沙箱我没有，原因是我没有『执行代码 / 写系统文件』这类工具——风险面小；
> 一旦加上就必须补，这比现在假装有更安全。」

一句话收尾（面试官最想听到的）：
> 从 Demo 到生产的差距，不在换更贵的模型，而在**是否认真处理了每一种「它可能不按预期工作」的时刻**。
