# GenAI Career Assistant · RAG + Agent 职业引擎

一个**可运行、可演示、可度量**的 GenAI 职业助手工程：左边是「LangGraph 多 Agent 职业助手」（教程 / 答疑 / 简历 / 面试题 / 模拟面试 / 职位搜索），右边是「企业知识库 RAG 引擎」（上传文档 → 混合检索 → 重排 → 带引用问答）。对外同时提供 **Streamlit 演示界面**与 **FastAPI 服务接口**，并内置 **效果评测体系**（Recall@5 / MRR / 命中率 / 幻觉率 + A-B 对比报告）。

> 目标：一份能直接写进简历、扛得住面试官追问（召回率多少？怎么优化的？为什么这么切分？）的工程化作品。

---

## 一、功能一览（7 个入口）

| # | 功能 | 模式 | 说明 |
|---|------|------|------|
| 1 | 教程生成 | `tutorial` | 联网检索后生成结构化 Markdown 教程（概念 + 可运行代码 + 参考链接） |
| 2 | 答疑问答 | `qa` | 资深 GenAI 工程师人设，多轮答疑，结束导出答疑记录 |
| 3 | 简历制作 | `resume` | 分步收集信息生成简历，支持基于 JD 改写 |
| 4 | 面试真题 | `interview_questions` | 检索整理目标岗位面试题清单 + 答案要点 |
| 5 | 模拟面试 | `mock_interview` | 面试官一问一答，结束输出结构化评价与改进建议 |
| 6 | 职位搜索 | `job_search` | 按城市 + 岗位关键词检索，整理为 Markdown 职位清单 |
| 7 | 知识库问答 | `knowledge` | 上传 PDF/Word/MD/HTML/TXT → 混合检索 + 重排 + 带引用问答 |

输入一句话会自动路由到对应模式，也可在侧边栏手动强制切换。支持通过 `user_context` 注入用户画像（由 jobseeker 薄网关传入岗位 JD + 关联简历），使 7 个能力直接结合用户背景作答。

---

## 二、快速开始

### 1. 准备环境与密钥

```powershell
# 安装依赖（Windows CPU，约几分钟）
python -m pip install -r requirements.txt
# 若 faiss 报 numpy 不兼容：pip install "numpy<2"

# 配置密钥
copy .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY（对话用）与 EMBEDDING_API_KEY（建库用）
```

> 缺 Key 也能启动：路由 / 建库 / 检索 / 评测界面均可演示，只是真实对话会提示「模型未配置」。

### 2. 启动演示界面（Streamlit）

```powershell
.\run_web.ps1        # 自动建 .venv 并装依赖；或手动 python -m streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### 3. 启动后端服务（FastAPI）

```powershell
.\run_api.ps1        # 或 python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# 接口文档 http://localhost:8000/docs
```

### 4. 跑评测并生成报告

```powershell
python run_eval.py --kb default --k 5
# 报告写入 Agent_output/Eval_Report_*.md
```

或在 Streamlit 侧边栏点「▶️ 运行评测并生成报告」。

> **评测提速**：默认 4 组实验 × 全量样本，LLM 调用较多。可用两个开关加速——
> `--no-judge` 跳过幻觉判定（省约 75% 调用）；`--workers 8` 提高按样本并发（默认 `EVAL_MAX_WORKERS=4`）。
> 幻觉率默认只在基线上判定一次并由各组复用（它衡量知识库内容能否支撑结论，与排序方式无关）。

### 5. Docker 容器化启动（推荐部署方式）

镜像为 CPU 版（`python:3.12-slim`，无 GPU、无 torch），同时编排 **API（:8000）** 与 **Web（:8501）** 两个服务，密钥经 `env_file` 注入、数据经 volume 持久化，**不写进镜像**。

```powershell
# 前置：本目录需存在 .env（已在，含百炼 Key）；如需干净分发先准备：
copy .env.example .env      # 再编辑填入 OPENAI_API_KEY / EMBEDDING_API_KEY

# 构建并启动（首次会拉取基础镜像 + 装依赖，稍慢）
docker compose up --build

# 仅起某服务（Web 走本地 graph，不依赖 API，可独立运行）：
docker compose up api       # 仅 FastAPI  → http://localhost:8000/docs
docker compose up web       # 仅 Streamlit → http://localhost:8501

# 后台运行 / 停止
docker compose up -d
docker compose down         # 默认保留 ./data 卷；加 -v 可一并清理数据
```

要点：
- **密钥不落镜像**：`.env` 已在 `.dockerignore` 排除，仅运行时由 `compose env_file` 注入，`config.py` 用 `os.getenv` 读取，与本地完全一致。
- **数据持久化**：`data` 用**命名卷** `app_data`（`data/index` 向量库、`data/knowledge` 文档、`app.db` 成本统计），重建容器不丢；产物 `Agent_output` 用 bind mount 直接映射到宿主机，方便查看与取用。
- **模型切换**：改 `.env` 的 `OPENAI_BASE_URL` / `MODEL_NAME`（如百炼 `https://dashscope.aliyuncs.com/compatible-mode/v1` + `qwen-plus-2025-07-28`）即可，无需改镜像。
- **探活与自愈**：两服务均带 `healthcheck` + `restart: unless-stopped`，异常自动重启。
- **非 root + 多阶段构建**：镜像以 uid 10001 的 `appuser` 运行；依赖装在 builder 阶段的独立 venv，最终镜像只拷贝 venv，不带 pip 缓存与构建残留——镜像更小，且容器逃逸也不会直接拿到 root。
  > Linux 宿主机若 `Agent_output` 写入报权限错误，执行一次 `sudo chown -R 10001:10001 Agent_output`。

> 更多配置见仓库根目录 `Dockerfile` 与 `docker-compose.yml`。

---

## 三、知识库 RAG 链路

```
文档 → loaders 解析 → clean 清洗 → split 切分
     → Embedding（云端 /embeddings）→ FAISS 向量库
     → BM25 稀疏库
提问 → 向量检索 + BM25 → RRF 融合 → Rerank（LLM / API / 关闭）→ 带引用生成
```

- **混合检索**：BM25 负责关键词精确匹配（岗位名、技术栈、专有名词），向量负责语义召回；RRF（Reciprocal Rank Fusion）融合，无需调权重。
- **重排**：`LLMReranker`（用对话模型对候选打 0–10 分，零额外依赖）/ `APIReranker`（硅基流动 `bge-reranker-v2-m3`）/ `NoneReranker` 三组可 A-B 对比。
- **多知识库隔离**：`src/rag/registry.py` 管理创建、列表、切换、删除、版本与统计；索引持久化到 `data/index/`，进程重启不重建。
- **引用溯源**：每条回答标注来源文件 + 片段编号，可展开查看原文，降低幻觉、便于核验。

### 评测指标口径（固定，保证报告可信）

- `Recall@5`：golden 片段是否出现在召回前 5 条，取全评测集均值。
- `MRR`：golden 片段首次出现排名的倒数均值。
- `Hit Rate@K`：前 K 条中至少命中一条的比例。
- `Hallucination Rate`：LLM-as-judge 将回答拆为断言，逐条判断是否可由召回片段支撑，不可支撑占比。

评测集支持从知识库 chunk 反向生成（`src/eval/dataset.py`）+ 人工校对，写入 `data/eval/eval_set.jsonl`。

---

## 四、工程化能力

- **LangGraph 工作流**：分类 → 路由 → 叶子节点（含 `fallback` 兜底），`src/graph/` 只做分类与路由。节点分支（结构化分类、关键词兜底、叶子节点、路由）均确定可测，`test_nodes.py` 已将其覆盖率推至 98%。
- **去阻塞化**：原 Notebook 的 `while True + input()` 改为 `SessionManager.start/step/finish` 单步推进，Web 与 API 共用。
- **流式输出**：界面默认**在对话气泡内逐字渲染**，不用干等整段生成完。链路是模型侧 `llm_stream` → `SessionManager.start_stream/step_stream` → SSE `/chat/stream`。实现要点：提交输入只做「入队」，真正的生成放到对话容器内部执行，因此流式文字直接落在助手气泡里，不会先在容器外渲染、结束再跳进消息列表。侧边栏有「⚡ 流式输出」开关，可切回一次性显示做对比；流式中途失败给友好提示而非抛栈；完整回复在流结束后才入栈，避免把「半截回答」带进下一轮上下文。
- **结构化输出**：分类结果、面评、职位清单、引用片段均用 Pydantic + `with_structured_output`。
- **稳定性**：`tenacity` 重试退避、显式 timeout、信号量限流、`diskcache` 结果缓存、降级链（联网失败→纯 LLM；Rerank 失败→按融合分）。
- **成本治理**：`tiktoken` token 统计与成本汇总，界面侧边栏实时展示。
- **安全与日志**：输入敏感词过滤、输出内容安全校验；统一日志仅记录 query 摘要（截断 120 字）/ 耗时 / token / 命中数，**禁止记录 API Key 与文档全文**。
- **多模型可切换**：DeepSeek（默认）/ 通义 / OpenAI 等，通过 `OPENAI_BASE_URL` 切换；Embedding 与 Rerank 同样可配置。当前 `.env` 使用百炼 `qwen-plus-2025-07-28`（与 `qwen-plus` 同系列的固定快照版，产出稳定、成本友好）。

### 可观测性：OpenTelemetry 链路追踪
- **开箱即用、零侵入**：未安装 `opentelemetry` 或未设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 时，全部埋点为 no-op，对业务零开销、零报错（见 `src/telemetry.py`）。
- **启用**：`pip install -r requirements-otel.txt`，再设置 `OTEL_EXPORTER_OTLP_ENDPOINT`（兼容 OTLP 的后端，如 Jaeger / Tempo / 阿里云 ARMS），进程启动即自动导出 span。
- **覆盖链路**：`server.request`（FastAPI 自动埋点）→ `agent.respond`（单轮生成）→ `graph.classify`（路由分类）→ `rag.retrieve` / `rag.ingest` / `rag.answer` → `llm.invoke` / `llm.stream`（模型调用，含 model / 字符数 / 错误类型 / latency）。`GET /health` 触发 `llm.health` span，可用于验证链路连通。
- **隐私**：span 只记录查询长度与模式，不记录用户输入原文与知识库片段内容。

### 代码规范与静态检查
- **统一规则**：`pyproject.toml` 固定 ruff 规则集（`E` / `W` / `F` / `I` / `UP` / `B`），行宽 120、目标 Python 3.10+；不依赖 ruff 默认集，避免版本升级造成本地与 CI 结果不一致。
- **本地提交前**：`pip install pre-commit && pre-commit install`，提交时自动修未使用导入 / 导入排序 / 旧式注解，并用 `detect-private-key` 兜底防止 `.env` 误提交。
- **CI 门禁**：`ci.yml` 中 `lint` 任务独立执行 `ruff check .`，与 `test` 任务（3.10 / 3.11 / 3.12 矩阵）并行。
- **豁免说明**：`src/prompts/*` 不限制行长（提示词按语义成行，折行会改变 prompt 内容）；`src/api/routers/*` 允许 `B008`（FastAPI 依赖注入惯用法）。
- **当前状态**：全仓库 `ruff check` 通过，0 违规。

### 测试与覆盖率
- **一键运行**：`pip install -r requirements-dev.txt` 后直接 `pytest`（`pyproject.toml` 中已配好 `testpaths` 与 `pythonpath`，不依赖调用方式）。
- **用例规模**：**186 条**（pytest 实际收集数，含参数化展开），分散在 18 个测试文件中，全部离线可跑——RAG / 评测用确定性伪 Embedding，接口用例在无 Key 时走降级路径，因此 CI 无需任何 API Key。
- **覆盖率**：`pytest --cov=src --cov-report=term-missing` 当前 **89%**（其中 `src/eval/runner.py` 已达 **100%**），CI 以 `--cov-fail-under=75` 作为门禁。
- **测试文件清单**：
  - `test_api.py`（接口鉴权 / 限流 / SSE / 知识库检索）
  - `test_graph_route.py`（7 模式路由与端到端冒烟）
  - `test_nodes.py`（LangGraph 节点与路由分支，本次新增，推高 `nodes.py` 至 98%）
  - `test_rag.py` / `test_loaders.py` / `test_rerank.py`（解析、混合检索、重排）
  - `test_eval.py`（评测指标与数据集生成）
  - `test_session.py` / `test_storage.py`（单步会话推进、产物存取与越权防护）
  - `test_hitl.py`（人机协同草稿→定稿全流程）
  - `test_telemetry.py`（链路追踪 no-op 兜底）
  - `test_safety.py`（注入拦截与脱敏）
  - `test_cache.py` / `test_tools.py` / `test_llm.py` / `test_embeddings.py`（缓存、工具、模型调用、Embedding）
  - `test_logging_setup.py`（日志脱敏与配置）
  - `smoke_test.py`（整体冒烟）
- **覆盖范围**：7 模式路由与端到端冒烟、FastAPI 鉴权 / 限流 / SSE / 知识库检索、内容安全（注入拦截与脱敏）、产物存储（含删除越权防护）、缓存与成本统计、文档解析失败降级、链路追踪 no-op 兜底、人机协同草稿→定稿全流程。
- **补全测试时捕获并修复的真实缺陷**：`src/eval/dataset.py` 原用 `model_dump_json(ensure_ascii=False)`，pydantic 2.11 起不再接受该参数，保存评测集会直接崩溃（已改为 `json.dumps(..., ensure_ascii=False)`）。

### 人机协同（Human-in-the-Loop）
高风险场景**不全自动化**——产物先落草稿，经人工确认后才定稿。

- **触发条件**：场景打标（`resume` / `mock_interview` / `job_search`）或用户输入命中高风险话题（薪资谈判 / 离职仲裁 / 背调 / 个人隐私）。
- **两阶段产物**：`finish()` 先落 `Resume_draft_时间戳.md` 并置 `awaiting_confirmation`，`confirm()` 才落 `Resume_时间戳.md` 并置 `finished`。草稿与定稿并存，保留「审阅前 / 审阅后」的审计证据。
- **状态透传**：`ChatResponse` 新增 `requires_confirmation` / `awaiting_confirmation` / `draft_artifact` 三个字段（均有默认值，向后兼容），SSE `done` 事件同步透传。
- **确认入口**：API `POST /chat/confirm`；Streamlit 在草稿态渲染红色警示条与「✅ 我已审阅，定稿」按钮。
- **只打标不拦截**：与 `safety.py` 的拦截机制职责分离——内容照常生成，只是定稿前需人看一眼，避免「需确认」被误做成「拒绝服务」。
- **可关闭**：`.env` 中 `ENABLE_HUMAN_CONFIRM=false` 即恢复全自动直接定稿，便于对比演示。

### 依赖锁定与供应链安全
- **锁定文件**：`requirements.lock.txt`（103 个包）与 `requirements-dev.lock.txt`（110 个，含开发依赖），由 `pip-compile` 编译生成，钉定全部**传递依赖**的精确版本。
- **为什么需要**：顶层声明的宽松约束（如 `pydantic>=2.7`）挡不住传递依赖在任意时刻升级。本项目真实踩过一次——pydantic 2.11 起 `model_dump_json(ensure_ascii=)` 被移除，保存评测集直接崩溃；锁定后这类「某天突然升级就崩」的问题变得可复现、可预期。
- **更新方式**（改完 `requirements.txt` 后重新编译）：
  ```powershell
  pip install pip-tools
  pip-compile requirements.txt      -o requirements.lock.txt     --no-emit-index-url
  pip-compile requirements-dev.txt  -o requirements-dev.lock.txt --no-emit-index-url
  ```
  `--no-emit-index-url` 必须带，否则会把本地镜像地址写进锁文件，导致 CI 装不上。
- **CI 保障**：`lock-verify` 任务固定 Python 3.12，用锁定版本安装并跑核心用例；矩阵任务仍用宽松依赖，专门验证跨版本兼容性——两者互补（锁文件按 3.12 编译，部分包如 `faiss-cpu` 在 3.10 未必有 wheel）。
- **漏洞扫描**：`audit` 任务用 `pip-audit` 扫描锁定依赖，**只告警不阻断**，避免上游 CVE 公告卡住日常开发。
- **容器构建**：`Dockerfile` 用锁文件安装，镜像内依赖完全可复现。

### 6 项 Agent 评估标准自检

| 标准 | 状态 | 落点 |
| --- | --- | --- |
| 真实业务 | ✅ | 7 个求职场景，目标用户明确，评测集可量化效果 |
| 后端工程 | ✅ | FastAPI + 鉴权 / 限流 / SQLite / 统一日志（脱敏）/ 全局异常处理 / SessionManager 多用户并发 |
| 核心能力 | ✅ | 任务拆解（LangGraph）、工具调用（联网 + 知识库）、容错（重试 + 降级链）、结果校验（结构化输出 + 安全过滤） |
| 上下文工程 | ✅ | 任务 / 记忆 / 知识库 / 工具结果，以及外部 `user_context`（jobseeker 网关传入的岗位 + 简历画像）按场景注入 prompt，非仅聊天记录 |
| 可观测性 | ✅ | OTel 全链路 span + 速度 / 成本 / token 统计 + 评测（Recall@K / MRR / HitRate / 幻觉率） |
| 人机协同 | ✅ | 高风险场景草稿→人工确认→定稿（见上一小节） |

---

## 五、目录结构

```
src/
├── config.py / llm.py / embeddings.py / models.py / cache.py / safety.py
├── state.py / storage.py / session.py / tools.py / logging_setup.py / telemetry.py
├── prompts/        # 分类 few-shot、7 个人设、RAG 提示词
├── rag/            # loaders/clean/split/vectorstore/bm25/fuse/rerank/pipeline/registry
├── agents/         # learning/interview/resume/jobsearch/knowledge + base
├── graph/          # nodes.py / workflow.py（LangGraph 路由）
├── api/            # db/schemas/deps/routers(main, chat, knowledge, sessions, health)
└── eval/           # dataset/metrics/runner/report
app.py              # Streamlit 入口
run_web.ps1 / run_api.ps1 / run_eval.py
tests/              # 18 个测试文件（test_api / test_graph_route / test_nodes / test_rag /
                   #   test_loaders / test_rerank / test_eval / test_session / test_storage /
                   #   test_hitl / test_telemetry / test_safety / test_cache / test_tools /
                   #   test_llm / test_embeddings / test_logging_setup / smoke_test）
```

---

## 六、API 速览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 同步对话（支持 `user_context`：外部注入的用户画像，每轮以 SystemMessage 前置） |
| POST | `/chat/stream` | **SSE 流式对话**（逐段返回 delta；同样支持 `user_context`） |
| POST | `/chat/finish` | 结束会话并导出产物 |
| POST | `/chat/confirm` | 人机协同：草稿确认定稿 |
| POST | `/knowledge/{kb}/ingest` | 上传文件建库（multipart） |
| GET  | `/knowledge` | 知识库列表与统计 |
| POST | `/knowledge/{kb}/search` | 知识库检索 |
| DELETE | `/knowledge/{kb}` | 删除知识库 |
| GET  | `/sessions` | 会话列表 |
| GET  | `/sessions/{id}` | 会话详情（含逐条消息，供外部归档） |
| GET  | `/health` | 健康检查 + 模型连通性探测 |

> **注意**：这些路径**没有 `/api` 前缀**（旧版文档写成 `/api/chat` 是错误的，已修正）。
> `chat` / `knowledge` / `sessions` 三组接口带鉴权，需请求头 `X-API-Key`（`API_KEY` 为空时关闭鉴权）。

详见 `http://localhost:8000/docs`。

---

## 七、常见问题（Windows）

- **faiss 导入报错 / `Symbol not found`**：多为 numpy 2.x 不兼容，执行 `pip install "numpy<2"`。
- **uvicorn 提示 uvloop 不支持 Windows**：属正常现象（Windows 无 uvloop，自动退化为默认事件循环），不影响功能。
- **GBK 解码错误**：所有文件读写已强制 `utf-8`，若仍报错检查 `.env` 是否为 UTF-8 编码。
- **建库无内容**：扫描件 PDF / 加密文档无法解析，请换可复制文本的文档。

---

## 八、面试加分点（详见 `INTERVIEW_NOTES.md`）

- 为什么选混合检索 + RRF？召回率提升多少？
- 为什么用 LLM Rerank 而非 CrossEncoder（无 GPU 约束）？
- chunk_size / overlap 怎么定？标题感知切分的收益。
- Function Calling 与显式检索两条路径的取舍。
- 评测集怎么来的（不是拍脑袋），指标口径如何固定。
- 覆盖率 89% 是怎么达成的？哪些分支最难测（如 OTel 真实 tracer 路径需 SDK）？

---

## 九、验证状态与已知问题

### 冒烟验证（playwright + 接口）
- Streamlit 界面完整渲染：品牌区、侧边栏（会话/知识库/检索参数/评测/成本）、示例芯片、产物与评测面板均正常。
- 自动路由可用：示例「帮我写一篇 LangGraph 实战教程」正确路由到「教程生成」模式并生成会话记录（写入产物面板）。
- 异常优雅降级：模型 Key 无效时，对话返回中文提示「模型调用失败，请检查 .env 配置后重试」，**不出现堆栈白屏**；缺 Key 时侧边栏显示「模型未连通」提示条。
- FastAPI 路由已注册并可用：`GET /sessions`、`GET /health` 均返回 200 与真实数据（经 `openapi.json` 核实含 `/chat`、`/knowledge`、`/sessions`、`/health` 等业务路径）。
- 测试套件：186 条用例、覆盖率 89%、`ruff check` 0 违规，CI 矩阵（3.10/3.11/3.12）全绿。

### 已知问题 / 注意
- **必须配置有效 Key**：对话与知识库建库分别依赖 `OPENAI_API_KEY` 与 `EMBEDDING_API_KEY`（Embedding 默认走硅基流动 `BAAI/bge-m3`）。未配置或无效时仅能演示路由/建库/检索流程，真实生成会降级。
- **Starlette 版本锁定**：`requirements.txt` 已固定 `starlette<1.0`——Starlette 1.x 会让 FastAPI 的 `include_router` 失效、导致全部业务路由丢失。安装后若 `openapi.json` 路径为空，请确认 starlette 版本。
- **Windows / faiss**：若 `import faiss` 报 numpy 不兼容，执行 `pip install "numpy<2"`。
- **OpenTelemetry 真实链路需安装 SDK**：运行环境未装 `opentelemetry` SDK 时，`telemetry.py` 走 no-op 兜底（已覆盖）；启用真实 span 导出需 `pip install -r requirements-otel.txt` 并配置 `OTEL_EXPORTER_OTLP_ENDPOINT`——这是当前 89% 覆盖率的主要缺口所在。
- 评测（`run_eval.py` 或界面「运行评测」）需要至少一个已建库的知识库，评测集会优先从知识库 chunk 反向生成。
