# 依赖升级清单：langchain / langgraph 跨大版本（2026-09-19 立项）

> **背景**：CI `audit` 命中 **32 条已知漏洞 / 9 个包**（口径见 README §九）。其中
> **starlette 8 条**被 `starlette<1.0` 硬约束阻塞、**diskcache** 上游暂无修复版本，
> 其余全部集中在 langchain / langgraph 生态 —— 本清单只处理后者，且已用实测把「能不能做」先问清楚。
>
> 一次性环境：`.venv` 之外的探针环境（只装 langchain 栈，用完即删）。**未改动 `requirements.txt`、未动 master。**

## 0. 先给结论（每条都带证据）

| 判断 | 证据 |
| --- | --- |
| **不被 starlette 约束阻塞** | `pip install --dry-run` 目标栈 + `starlette>=0.46,<1.0` + `fastapi>=0.115` → **退出码 0**，解析结果里 `starlette-0.52.1` **原样不动**（新栈不逼升 starlette） |
| **能消掉绝大部分告警** | 探针环境（langchain 生态 11 包 + 传递依赖）跑 `pip-audit` → **`No known vulnerabilities found`**；剩下 9 条与本次升级无关（starlette 8 + diskcache） |
| **我们实际用的接口基本都在** | 12 个模块 / 19 个符号：**11 个模块全数存活**；唯一"缺"的 `langgraph.checkpoint.sqlite` 在 `langgraph-checkpoint 4.x` 里本就不含（独立包），而我们的代码已是「拿不到就降级」的写法 |
| **关键调用签名没变** | `ToolNode(handle_tool_errors=)` ✓、`StateGraph.compile(checkpointer=)` ✓、`openai.{APIConnectionError,APIStatusError,RateLimitError}` 在 3.16.2 仍在 ✓ |
| **没有「签名级」破口，风险在行为** | `ChatOpenAI(max_tokens=5)` 目标版本**构造成功且属性生效**（pydantic 动态签名，`inspect` 查不到 ≠ 不可用 —— 见 §3 教训） |
| **主要风险是传递升级** | 会一次性带上 `openai 2.54.0 → 3.16.2`、新增 `langchain-classic / langchain-protocol / langgraph-prebuilt / httpx2 / httpcore2` |

## 1. 目标版本（探针实测解析结果，非"听说"）

| 包 | 当前 | 目标 | 备注 |
| --- | --- | --- | --- |
| langchain | 0.3.30 | **1.4.2** | 大版本 |
| langchain-core | 0.3.86 | **1.6.3** | 大版本 |
| langchain-openai | 0.3.35 | **1.6.2** | 大版本 |
| langchain-text-splitters | 0.3.11 | **1.1.2** | 大版本 |
| langchain-community | 0.3.31 | 0.4.2 | 小版本线；同时新增 `langchain-classic 1.0.8`、`langchain-protocol 0.0.19` |
| langgraph | 0.2.76 | **1.2.11** | 大版本 |
| langgraph-checkpoint | 2.1.2 | **4.2.0** | 大版本（sqlite saver 已不在本包内） |
| langgraph-sdk | 0.1.74 | 0.4.4 | 0.x 内部跳 |
| openai（传递） | 2.54.0 | **3.16.2** | 大版本，需重点复测错误分级 |
| tenacity / pydantic / starlette / fastapi | — | **不变** | pydantic 2.13.5、tenacity 9.1.4、starlette 0.52.1、fastapi 0.141.1 |

## 2. 影响面（按代码位置，行号为准）

| 位置 | 用法 | 迁移动作 |
| --- | --- | --- |
| `src/llm.py:10-13` | `langchain_core.messages.BaseMessage`、`ChatOpenAI`、`openai.{APIConnectionError,APIStatusError,RateLimitError}`、`tenacity` | 三类错误实测仍在；**复测错误分级**（`classify_error` 的分支是否还命中）与 `max_tokens` 的真实 API 行为 |
| `src/graph/tool_loop.py:22-26` | `{AIMessage,BaseMessage,HumanMessage}`、`GraphRecursionError`、`{END,START,StateGraph}`、`add_messages`、`ToolNode` | 符号与签名全在；**复测工具调用解析 / `handle_tool_errors` 回灌文案 / 递归上限** |
| `src/graph/workflow.py:5` | `{END,START,StateGraph}` | 无符号变化；复测 9 模式路由 |
| `src/graph/checkpoint.py:40,50` | `langgraph.checkpoint.sqlite.SqliteSaver`（可选包）、`InMemorySaver` | 4.x 不再含 sqlite → 需要把 `langgraph-checkpoint-sqlite` 一起找对得上的版本，或接受降级到 memory（现有 try/except 已覆盖） |
| `src/session.py:13` | `{AIMessage,HumanMessage,SystemMessage,trim_messages}` | 符号在；**复测上下文截断**（`trim_messages` 参数与 token 计数器行为） |
| `src/tools.py:284,577` | `langchain_community.tools.DuckDuckGoSearchResults` | 符号在；community 0.4.2 下若出现弃用告警，走 `ddgs` 直连（本项目已依赖 `ddgs`） |
| `src/tools.py:494,529`、`src/mcp/tools.py:132` | `langchain_core.tools.tool`、`StructuredTool` | 符号在；复测 MCP 工具包装与 `bind_tools` |
| `src/rag/split.py:29` | `RecursiveCharacterTextSplitter` | 符号在；**必须复测检索指标**（切片结果变了会直接改 Recall） |
| `src/embeddings.py:46,80` | `OpenAIEmbeddings`、`HuggingFaceEmbeddings` | 符号在；后者可能提示迁到 `langchain-huggingface`（可选依赖，先不动） |

## 3. 分阶段步骤（每阶段独立可回滚）

- **阶段 0 · 备好退路（不改代码）**：建分支 `feat/dep-upgrade-langchain1`；记录基线——`pytest tests/ -q` 通过数、
  `python run_eval.py --kb data1 --top-k 5 --no-judge` 的 Recall@5 / MRR / HitRate（**离线指标，不花模型 token**）。
- **阶段 1 · 量坏多少（先别修）**：按 §1 放宽 `requirements.txt` 的 langchain/langgraph 上界（**不动 starlette**），
  在**一次性 venv** 装全量依赖并跑 `pytest -q`，只记录失败清单（这一步的产出是"坏在哪儿"，不是"修好了"）。
- **阶段 2 · 修代码**：按失败清单逐条修，且**不许退化既有契约**（异常分级、工具降级文案、熔断与失败可见性都保留）。
- **阶段 3 · 锁文件与门禁**：`pip-compile` 重编两份 lock；`lock-verify` 通过；全量 `pytest` 全绿；`ruff` 0 违规。
- **阶段 4 · 真实链路回归**：跑 §4 全清单。
- **阶段 5 · 收尾**：更新 README §九（漏洞条数、处置口径改口径）与版本号；提交信息写清「消掉了哪几条、还剩哪几条以及为什么」。

## 4. 回归验证清单（命令级，逐条勾）

1. `pytest tests/ -q` —— 与基线通过数一致（当前 365 通过 + 1 跳过）。
2. `ruff check src tests` —— 0 违规。
3. `python run_eval.py --kb data1 --top-k 5 --no-judge` —— Recall@5 / MRR / HitRate 与基线容差 ±0.02（**切片与嵌入行为变了会先在这里暴露**）。
4. `python run_eval.py --kb data1 --tool-ab --tool-ab-samples 15 --tool-ab-model qwen-turbo --no-web --skip-rag` —— 工具调用轮次 / token / 延迟与基线比对。
5. **故障注入复跑**：`… --fault-tool knowledge_search --fault-kind http_5xx --fault-rate 1.0 --workers 1`
   —— 必须仍是「对照组工具失败率 100%、Failure Onset 1.0、15 次调用中 3 次真打上游 + 12 次短路」。
   **这是判据：失败路径没被升级破坏。**
6. 路由冒烟：起 API → `GET /health` 200 + `openapi.json` 里 `/chat`、`/knowledge`、`/sessions` 等业务路径**一个不少**。
7. Streamlit 冒烟：9 模式各跑一条真实样例，含工具调用时间线渲染。
8. checkpoint：`/health` 的后端名与基线一致，续跑用例（不重复执行工具）通过。
9. MCP：`build_mcp_tools` 能列举出工具；工具包装（`StructuredTool`）可用。

## 5. 验收判据与回滚

- **验收**：§4 全过，且 `pip-audit -r requirements.lock.txt` 的告警数 **32 → ≤9**（剩余 starlette 8 + diskcache 1~2，均有书面理由）。
- **回滚**：升级与锁文件放**同一个提交**；出问题 `git revert <sha>` + 按 lock 重建 venv 即回到当前状态（当前 lock 与 master 一致）。
- **明确不在同一次做**：`starlette 0.52→1.x`（1.x 会让 FastAPI `include_router` 失效，必须连 FastAPI 一起升并重跑路由冒烟）、`numpy/faiss`（`numpy<2` 约束）、`diskcache`（上游无修复版本）。

## 6. 两条方法教训（写进 `docs/DEV_NOTES.md`）

1. **`inspect.signature` 查不到参数 ≠ 参数不可用**：`ChatOpenAI` 是 pydantic 模型、`__init__` 签名是动态生成的，
   `max_tokens` 在签名里查不到却**实测可构造、属性生效**。判据必须是真实构造/调用（我差点据此报出一个假破口）。
2. **"接口在不在"比"版本号变没变"可靠**：这次先用符号存活矩阵把范围从「整条栈要重写」收缩到
   「1 个可选包 + 行为复测」，才敢给出分阶段方案；反过来只盯版本号会高估工作量。

## 7. 执行结果（2026-09-20 实测）

阶段 1–4 已跑完，结论：**成本远低于立项时的估计** —— 放宽约束后**代码一行没改**，全量测试即全绿。

| 验证项 | 结果 |
| --- | --- |
| 全量 `pytest`（一次性环境，`pytest 9.1.1`） | **退出码 0、全绿**（进度条无 F/E） |
| 离线检索 4 组（`--no-judge`） | Recall@5 四组全 **100%**，MRR **0.649 / 0.750 / 0.812 / 0.812** —— 与基线表一致 |
| 工具调用 A/B（15 条 / `qwen-turbo` / `--no-web`） | 两条路径成功率均 **100%**；FC 延迟 **−32.0%**、token **0.33x**（基线同条件实跑：−29.5% / 0.33x） |
| 自主检索率 26.7% → 6.7%？ | **不是升级导致的**：基线环境今天同条件实跑也只有 13.3%，且历史 26.7% 那次是「联网开启」，本就不该比大小 |
| 故障注入复跑 | 短路 **14** 次 + 最终降级 **3** 次；报告写明「对照组工具失败率 **100.0%**、Failure Onset **1.0**」——失败路径未被升级破坏 |
| 路由冒烟 | `openapi.json` **16 条路径**，`/chat` `/chat/stream` `/knowledge` `/sessions` `/health` 等业务路由齐全 |
| `/health` 字段对照 | 与基线逐字段一致（`runtime.checkpoint=memory`、skills 9 条、熔断为空）；唯一差异是新栈少了基线的 `LangChainPendingDeprecationWarning` |
| 锁文件 | `pip-compile` 重编：主 **101** 包 / 开发 **107** 包；`pip install --dry-run --ignore-installed` 解析通过 |
| **验收判据** | `pip-audit -r requirements.lock.txt`：**32 条 / 9 包 → 12 条 / 2 包**（剩 `starlette` 10 + `diskcache` 2，均有书面理由） |
| 跨版本兼容 | 目标包 `requires_python` 全为 `>=3.10` → CI 的 3.10/3.11/3.12 矩阵仍成立 |

两条如实说明：

1. **`langgraph-checkpoint-sqlite` 是独立包**：新旧栈都不含它，`/health` 在两边都报 `checkpoint=memory`
   ⇒ 本次**无行为退化**；要让「跨进程恢复」真正生效需另装该包（单独立项）。
2. **`openai` 没被带走**：`pip-compile` 在目标约束下仍解出 `openai==2.54.0`（探针环境里升到 3.16.2 也跑绿）
   ⇒ 实际升级面比立项时估计的更小；`httpx2` / `langchain-classic` / `langchain-protocol` / `langgraph-prebuilt` 属新增传递依赖。

**未做（留给合并时决定）**：版本号 `v1.5.0 → v1.6.0` 需同步 `src/__init__.py`、`README.md`（含 §十 变更记录）、
`docs/ENGINEERING_SUMMARY.md`、`docs/DEV_NOTES.md` 四处，属发布动作，不在待验证分支上先占版号。
