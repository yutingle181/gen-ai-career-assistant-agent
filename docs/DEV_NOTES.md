# Agent实习项目 开发记录（DEV_NOTES）

> 本文件由 `preflight-check` skill 维护：动手前先读本文件，踩坑后追加记录。
> 生成时间：2026-09-19 23:24
> 重新生成：`dev-notes.ps1 -Init -Force`；追加记录：`dev-notes.ps1 -Add`

## 一、项目速览

- 项目名称：Agent实习项目
- 版本号：1.6.0
- 技术栈：Python · FastAPI · Streamlit · LangGraph/LangChain（RAG：faiss + BM25）
- 仓库根目录：D:\project\Agent实习项目

## 二、常用命令

| 用途 | 命令 |
|---|---|
| 编译 / 构建 | 无需编译（纯 Python）；装依赖：`pip install -r requirements.txt`（锁定版 `-r requirements.lock.txt`） |
| 测试 | `python -m pytest -q`；lint：`python -m ruff check src tests` |
| 本地启动 | Web：`python -m streamlit run app.py`（或 `.\run_web.ps1`）；API：`python -m uvicorn src.api.main:app --port 8000`（或 `.\run_api.ps1`）；评测：`python run_eval.py --kb default --top-k 5` |
| 提交前自检 | `powershell -ExecutionPolicy Bypass -File <preflight-check skill>/scripts/preflight.ps1 -Path "D:\project\Agent实习项目"` |

## 三、端口与运行环境

- Streamlit Web：`8501`（`.\run_web.ps1` 或 `python -m streamlit run app.py`）
- FastAPI（uvicorn）：`8000`（`.\run_api.ps1`，接口文档 `/docs`）
- Python 3.12；CI 同时跑 3.10 / 3.11 / 3.12 矩阵

## 四、目录与配置约定

- `.codebuddy/`
- `.github/`
- `.idea/`
- `.playwright-cli/`
- `.playwright-mcp/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.venv-ci/`
- `.vscode/`
- `__pycache__/`
- `Agent_output/`
- `data/`
- `docs/`
- `src/`
- `tests/`

## 五、本项目已发现的"偏离默认"约定

> 记录与框架/工具默认行为不同、容易踩坑的地方（如构建产物输出到非默认目录、被忽略的目录、必须额外指定的配置文件）。

- 以下路径已被 .gitignore 忽略，提交时不会带上（也不要强行 add）：__pycache__/、.eggs/、build/、dist/、.pytest_cache/、.mypy_cache/、.ruff_cache/、.venv/、.venv-ci/、venv/、env/、ENV/。
- **`starlette` 必须 `<1.0`**：1.x 会让 FastAPI 的 `include_router` 失效、业务路由全丢（`openapi.json` 会变成空路径）。升级依赖时先看 `openapi.json` 里的路径数。
- **D 盘有「安全删除」策略**：批量删除会被拦截（提示 `SAFE_DELETE_BULK_CONFIRM_REQUIRED`），需人工确认；临时目录/一次性虚拟环境建议建在 C 盘或当前工作区内，避免留垃圾删不掉。
- **CI 的 `audit` 任务只告警不阻断**：步骤内自己吞掉 `pip-audit` 的非零退出，命中时打 `::warning` 并把明细写进 Job Summary（`continue-on-error` 只是兜底）。**它报红不代表业务失败**，处置口径见 README §九。
- **`master` 已开分支保护（禁强推 / 禁删除）**：2026-09-20 起生效——改代码一律走「分支 → PR → 等 CI 绿 → 合并」，别直接往 master 提交。**代价是 master 不能再 `git push --force`**：若将来需要再次改写历史（如清除误提交的隐私文件），**必须先到 `Settings → Rules`（旧界面在 `Branches`）临时关闭该规则**，改完再开回。自查是否生效：仓库首页若还出现「Your master branch isn't protected」提示，就是没开。
- **可选依赖要在开发锁里装齐**：`langgraph-checkpoint-sqlite` 属**生产可选**（未装自动降级为内存 checkpointer），但 `requirements-dev.txt` **必须**声明——否则跨进程恢复的用例会整体 skip，「恢复能力到底可不可用」永远没有测试兜底。改完 `requirements*.txt` 记得重编锁并补哈希（哈希格式见踩坑 4）。

## 六、踩坑记录

> 追加命令：`dev-notes.ps1 -Add -Title "标题" -Symptom "现象" -Cause "原因" -Fix "解决"`
> 每条记录包含：日期 / 现象 / 原因 / 解决 / 标签。

<!-- 新记录由脚本自动追加到本行下方 -->

### 1. 提交报 empty commit message 但消息文件其实是好的

- 日期：2026-09-19
- 现象：git commit -F 「中文路径下的消息文件」报 Aborting commit due to empty commit message（退出码 1）；同一写法在上一次提交是成功的，且该文件由文件工具写入并报成功
- 原因：未定论。已用对照实验排除三项猜测：① 中文路径——在一次性 git 仓库用完全相同的构造提交成功、中文无损；② 跨工作区写入——文件工具对工作区外仓库写入正常（29 字节、内容正确）；③ -F 读取构造本身——复现通过。仅剩「那一次文件落盘/读取时序」未能复现，故不写死结论
- 解决：消息文件改放 ASCII 路径（工作区内）后一次成功；并加两道校验：提交前确认消息文件字节数非 0、提交后用 git log -1 --format=%s 核对主题；若再报空，改用 -m 或 git commit --amend -F 兜底
- 标签：git,powershell,提交信息,归因

### 2. 用 inspect.signature 判断第三方库参数是否可用会误判（pydantic 动态签名）

- 日期：2026-09-19
- 现象：依赖升级预检时，判断 max_tokens 是否在 ChatOpenAI 初始化签名里得到 False，据此差点得出「升级必破」的结论；实测构造 ChatOpenAI(model=..., api_key=..., base_url=..., max_tokens=5) 成功且属性生效，max_completion_tokens 指向同一字段
- 原因：LangChain 系模型类是 pydantic BaseModel，初始化签名由 pydantic 动态生成，字段可以别名或额外关键字形式接受，签名里未必逐字出现
- 解决：兼容性判据用真实构造 / 真实调用并断言关键属性，签名只作线索；符号是否存在用 hasattr、参数是否可用用调用、行为是否等价用跑测试；对 pydantic 系库尤其如此
- 标签：python,pydantic,依赖升级,验证方法

### 3. 跨环境验证踩的两个坑：PowerShell 管道带 BOM / 脚本的 sys.path

- 日期：2026-09-20
- 现象：① 用 here-string 管道把代码传给 python（python -）报 SyntaxError: invalid non-printable character U+FEFF；② 用绝对路径运行临时脚本（cwd 已是项目根）报 ModuleNotFoundError: No module named src
- 原因：① PowerShell 5.1 的管道/输出编码会给内容加 UTF-8 BOM，Python 把 BOM 当成源码首字符；② 脚本模式下 sys.path 第一项是脚本所在目录而非当前工作目录，所以 cwd 在项目根也 import 不到包
- 解决：① 代码写进临时文件（[IO.File]::WriteAllText + UTF8 无 BOM）再按文件路径执行；② 设 env:PYTHONPATH 指向项目根，或把临时脚本放进项目内执行后删除
- 标签：powershell,python,编码,跨环境验证

### 4. 手写 --hash 行漏掉反斜杠续行，pip 会静默忽略全部哈希

- 日期：2026-09-20
- 现象：给锁文件补哈希后，pip 在 --require-hashes 下报 3011 条 WARNING: line N has --hash but no requirement, and will be ignored，随后 ERROR: Hashes are required in --require-hashes mode；即每个哈希都被当成没有需求的行丢弃
- 原因：pip-compile 的哈希格式是反斜杠续行（需求行以「反斜杠」结尾，后续缩进的 --hash 行才算它的续行）。只把哈希写成缩进行、不加行尾反斜杠时，pip 不会把它们挂到上一条需求上
- 解决：生成脚本改为「需求行 + 行尾反斜杠」，最后一哈希行不加；判据：格式类警告数与「Hashes are required」报错数必须为 0（网络类警告无关）
- 标签：pip,锁文件,供应链,格式

### 5. sqlite saver 的连接被垃圾回收关掉，checkpoint 静默降级成内存

- 日期：2026-09-21
- 现象：已装 `langgraph-checkpoint-sqlite`、日志也报 `backend=sqlite`，但跨进程续跑始终无效；后续任何读写都报 `Cannot operate on a closed database`
- 原因：`SqliteSaver.from_conn_string()` 返回的是**上下文管理器**，`__enter__()` 取出 saver 后若把 cm 丢掉，它被垃圾回收时会触发 `__exit__` 关闭连接
- 解决：自己建连接（`sqlite3.connect(path, check_same_thread=False)` + `SqliteSaver(conn)`）并把连接与 saver 放进 `_KEEPALIVE` 保活，`reset_checkpointer()` 时才释放；同时新增 `_probe_saver()` 可用性自检——**import 成功不等于版本兼容**，正是它把这个问题从「静默降级」变成「日志可见」
- 判据：`gc.collect()` 之后 saver 仍能读写（`tests/test_checkpoint_resume.py::test_sqlite_saver_survives_garbage_collection`）
- 标签：langgraph,sqlite,checkpoint,GC,可选依赖

### 6. `get_state()` 会合并 pending_writes，用 `snapshot.next` 判断「轮次是否结束」会误判

- 日期：2026-09-21
- 现象：进程在工具节点执行中途硬崩（`os._exit(9)`）后，同一 thread 续跑**返回空字符串**——模型与工具一次都没再跑，用户拿到空答案
- 原因：`app.get_state()` 会把 `pending_writes` 合并进 state，`snapshot.next` 因此变成空元组；按「next 为空 = 本轮已完成」判断就会直接复用结果，可那一轮其实停在**未执行的工具调用**上（state 里只有用户消息）
- 解决：收束判据改为「state 中是否存在**非空 AIMessage**」（复用 `_final_text`）；未收束则 `app.invoke(None)` 纯续跑，未完成的工具节点会重跑一次（at-least-once）
- 判据：`tests/test_checkpoint_cross_process.py::test_resume_after_hard_crash_finishes_the_turn`（续跑输出非空 + 工具计数为 2）
- 标签：langgraph,checkpoint,pending_writes,续跑,at-least-once
### 7. Windows 上编译的锁文件会漏掉平台专属依赖，Linux 安装时在哈希模式报错

- 日期：2026-09-21
- 现象：CI（Ubuntu）安装 `requirements-dev.lock.txt` 报
  `ERROR: In --require-hashes mode, all requirements must have their versions pinned with ==. These do not: uvloop>=0.15.1 (from uvicorn[standard]==0.52.4 -> -r requirements-dev.lock.txt (line 2788))`；
  而 Windows 本地装同一份锁完全正常，锁里也「未钉版行 = 0」
- 原因：uvloop 不支持 Windows，`pip-compile` 在 Windows 上解析 `uvicorn[standard]` 时把这条依赖**整条略过**——
  锁里既没有 uvloop 的钉版行、也没有它的哈希。Linux 上 uvicorn 的 extra 依然要求 uvloop，
  而文件里一旦出现 `--hash`，pip 就**自动进入哈希模式**（不许临场补装）→ 直接失败。
  括号里的 `line 2788` 是父依赖 uvicorn 的行号，不是 uvloop 的位置，别被它带偏。
- 解决：按 Linux 编译会得到的样子补一条 `uvloop==<ver> ; sys_platform != "win32"`（带 PyPI 全平台文件哈希，含 linux 轮子）；
  **两份锁都要补**（主锁供镜像构建，开发锁供 CI 安装）。删掉该行也能让 CI 变绿，但那等于 Linux 运行时悄悄少一个加速件，属静默偏差。
- 判据：`python scripts/check_lock_platform_deps.py requirements.lock.txt requirements-dev.lock.txt`
  —— 用 PyPI 元数据把每个包的 Linux 依赖集展开、与锁做差集，**差集必须为空**（本次就是靠它确认「uvloop 是唯一缺口」，
  而不是反复推 CI 试出来）；另加在 Linux 上 `pip install --require-hashes -r requirements-dev.lock.txt` 成功（CI 的 lock-verify 已覆盖）。
- 注意：**这类问题在 Windows 上无法复现**（同一个锁在 Windows 上装得好好的），所以静态检查与 Linux 侧安装两条都要留着。
- 标签：pip,锁文件,跨平台,哈希模式,uvloop,供应链

### 8. 模型/向量免费额度耗尽，让 /health 静默退成 degraded（换模型前先探测）

- 日期：2026-10-15
- 现象：`/health` 从 ok 变成 `{"status":"degraded","llm":false,"embedding":false}`。服务照常起、接口照常 200，
  但模型与向量都调不动（回答走降级）。日志里只有一条 403：`AllocationQuota.FreeTierOnly`
  —— "Free quota exhausted… please add funds or disable the \"use free tier only\" mode"。
- 原因：`.env` 把模型名**写死**（`MODEL_NAME=qwen-plus-latest`、`EMBEDDING_MODEL=text-embedding-v3`），
  而这两个模型的免费额度恰好用尽。控制台里其它模型还各剩百万级 tokens ——**不是没额度，是钉错了模型**。
  最坑的是它不抛异常、不影响启动，只在 `/health` 里体现，不主动查就会以为"代码坏了"。
- 解决：新增根目录 `probe_models.py`（只用标准库，虚拟环境坏了也能跑）：逐个候选发**最小请求**
  （chat 用 `max_tokens=1`、embeddings 用 4 个字），输出「模型 → OK(HTTP 200) / FAIL(403|400 + 错误码 + 截断 message)」，
  **只报密钥长度不报值**，退出码 0/10/20（10=会话可用但无可用向量，20=无可用会话）。
  实测结论：会话侧 7 个候选全 200（deepseek-v4.1-flash / qwen3.8-flash / qwen3.8-27b / qwen3.8-max-0902 /
  glm-5.3 / kimi-k3 / deepseek-v4-flash-0731），仅当前钉的 `qwen-plus-latest` 403；
  向量侧 `text-embedding-v3/v4/v2` 全部 400 额度耗尽、`qwen3-embedding-*` 404 或 Access denied、
  `multimodal-embedding-v1` 该兼容模式不支持 → 向量侧改走 `EMBEDDING_PROVIDER=local`（BAAI/bge-small-zh-v1.5）。
- 判据：`GET /health` 回到 `status=ok` 且 `llm/embedding=true`；探测原始结论留在 `data/eval/probe_models.json`。
- 注意：
  - **换向量模型或维度后必须重建 faiss 索引**（`run_eval.py --rebuild`），否则维度不匹配直接检索失败，
    这是最容易漏、报错最莫名其妙的一步；
  - 按厂商生效的优化要重新评估：`PROMPT_CACHE_MODELS=("qwen",)` 意味着 prompt 缓存对 deepseek 系不生效；
  - `cache.py` 的单价表未收录的模型按 0 计（有测试契约钉住，不改行为），但现在会**告警一次**，
    否则"0 成本"会被误当成"不花钱"。

### 9. 模型路由把厂商写死，换成 deepseek 后快模型静默失效

- 日期：2026-10-15
- 现象：为了换额度把 `MODEL_STRONG` 改成 deepseek 系之后，`MODEL_FAST=qwen3.8-flash` **永远不被选中** ——
  不报错、不打日志，只是所有请求都走强模型，分级路由的收益悄悄归零。
- 原因：`select_model()` 里有一句 `if not config.MODEL_STRONG.lower().startswith("qwen"): return MODEL_STRONG`。
  这是 9.2.A 当时"生产 qwen-plus / qwen-turbo 双模型部署"的假设，换厂商后前提直接不成立。
- 解决：抽 `_routing_available()`：开关开 + `MODEL_FAST` 非空 + 与 `MODEL_STRONG`（去空格、忽略大小写）不是同一个模型
  即可降级，**跨厂商同样生效**；其余保守分支全部不动（带 `user_context` / 非 tutorial,qa 模式 /
  复杂关键词 / 超长 query / 任何异常 → 一律回落 STRONG）。`tests/test_model_routing.py` 用三条断言钉住：
  跨厂商可降级、快模型缺失或与强模型同款时不降级、路由内部异常回落 STRONG。
- 判据：`pytest tests/test_model_routing.py` 全绿；实际请求日志里能看到两条路径分别用了哪个模型。
- 注意：这类"厂商硬编码"最隐蔽的地方是**文档也一起变成假话** —— `config.py` 注释与 `.env.example`
  原文都写着"仅 qwen 家族生效"，改代码不改文档，下一个人会照着错误前提做决策。
- 标签：模型路由,厂商耦合,静默失效

### 10. 换 PyPI 源：镜像"能访问"不等于"能装到包"

- 日期：2026-10-15
- 现象：`pip install sentence-transformers` 报
  `Could not find a version that satisfies the requirement sentence-transformers>=3.0 (from versions: none)`，
  而 `pip config list` 里明明配了国内镜像（清华 `.../pypi/web/simple`）。
- 原因：那个镜像地址对本机**取不到包清单**。而"能不能用"不能用根路径探测来判断 ——
  各镜像根路径的表现不一致（有的 200、有的 403），根路径可达 ≠ 包列表可读。
- 解决：逐个候选镜像用**真实包路径** `<index>/sentence-transformers/` 探测，取第一个能列出包的。
  本机实测可用：阿里云（505ms）、腾讯云（375ms）、USTC（404ms）、华为云（356ms）、官方 pypi（842ms）。
- 判据：pip 退出码 0，且 `import sentence_transformers` 成功、`SentenceTransformer(...).encode()` 能出向量。
- 附带：国内访问 huggingface.co 会超时，下载模型前先设 `$env:HF_ENDPOINT='https://hf-mirror.com'`
  （已写进 `requirements-optional.txt` 与 README，免得一换机器就踩）。
- 标签：pip,镜像,可选依赖,HuggingFace镜像

### 11. 「未改动代码也会失败」的测试才是真 flaky：跨进程硬崩续跑的工具执行次数

- 日期：2026-10-15
- 现象：`tests/test_checkpoint_cross_process.py::test_resume_after_hard_crash_finishes_the_turn`
  偶发失败：`AssertionError: 未完成的工具节点会重跑一次（at-least-once） assert 3 == 2` ——
  崩溃后续跑里工具多执行了一次。
- 排查：先用 `git stash` 把本轮全部改动 stash 掉再跑，**仍然失败**；再在未改动代码上重复 5 次，
  分布是 **3,2,2,2,3**（2/5 失败）；而逐个还原任何一个互不相关的文件都能让它"通过"。
  结论：与代码改动无关，是**时序敏感**（崩在哪一落盘点决定重放几个节点），
  任何"改一点就能过"的假象都是噪声——这也是为什么必须用"未改动代码 + 多次重复"来证伪。
- 原因：`os._exit(9)` 发生在**工具函数内部**，此刻 assistant 节点那条写入是否已提交到 SQLite
  不确定（页缓存 / WAL 落盘时机）。已落盘 → 续跑只重放 tools 节点，工具共执行 2 次；
  未落盘 → 连 assistant 一起重放，模型再产出一遍 tool_calls，工具共执行 3 次。
- 解决：把断言从 `== 2` 改成 `>= 2`，并把理由写进 docstring —— 原断言把
  "崩溃瞬间的落盘时机"误当成了系统语义，而本用例要守的语义是 **at-least-once**：
  ① 工具至少重跑一次（不被跳过）② 这一轮一定能收束出答案。真正"重发已完成轮次不重复执行"
  由同文件另一条用例守住（无崩溃、确定性强）。
- 判据：未改动代码上连续 5 次都通过；`git stash` 前后行为一致（不再"改点啥就过"）。
- 附带（同一个晚上踩的第二个坑）：多个本地实例**共用 `data/checkpoints.sqlite`** 时，
  后起的实例会在 checkpointer 建表上抢写锁而**卡死**（`/health` 一直超时，但进程活着、
  日志可能还打出过一个 200）—— 本地同时起多个实例时用 `CHECKPOINT_DB_PATH` 指向不同文件。
- 标签：flaky测试,at-least-once,跨进程,checkpoint,sqlite锁

### 12. 「自动恢复」写了却不会恢复：构造与探活耦合在一起

- 日期：2026-10-16
- 背景：给任务状态加 Redis 后端（多副本取消/背压共享），要求是「Redis 挂着自动回退进程内、
  恢复后自动切回」，实现为 `TaskStoreRouter`：构造 → 探活 → 选后端 → 每次操作前按间隔重试探活。
- 现象：单测 `test_router_recovers_when_redis_comes_back` 始终失败（期望切回 redis、实际仍是 memory），
  而启动期降级用例是通过的，看起来"降级正常"，所以第一反应是恢复逻辑写错了。
- 原因：**探活和构造被写在同一个 `try` 里**，探活抛的异常（ConnectionError）与构造失败走同一条
  `except` → `self._redis = None`。于是"Redis 启动时没连上"这一种最常见的情况下，
  路由器**永远拿不到实例**，恢复探测无从下手 —— 自动恢复只在「启动成功、中途才坏」时才成立。
  两次返工（第一次把 ping 从 `create_store` 里挪出来，但 `try` 仍把 `store.ping()` 包在里面 ✗）
  之后才改成：**构造与探活分开**（build 成功即保留实例；探活失败只决定用哪个后端）。
- 教训：
  1. **降级与恢复是两件事，只测降级会漏掉恢复** —— 必须成对写用例；
  2. 判据要选"**实例还在不在**"这种可观测事实，而不是只看当前 active 是谁；
  3. 这类"写了但不会执行"的分支，注释里写"会自动恢复"比不写更糟：文档承诺了代码没兑现的能力，
     下一个人会照着错误前提做部署决策。
- 附带（同轮修掉的另一处）：早期版本把"写取消状态位"和"广播"拆成两步调用，
  调用方一旦只做其中一步，就会出现"广播了但查不到状态位"这类只在多副本下暴露的不一致；
  现在 `mark_cancelled` 内部一次做完（先写后广播），广播失败只影响响应速度、不影响正确性。
- 标签：降级,自动恢复,Redis,多副本,单测设计

### 13. 降级判据写成「恰好是那个实例」，导致降级会静默失效

- 日期：2026-10-16
- 现象：写「Redis 中途挂掉时各操作都要降级」的用例时，`save` 明明抛了
  `ConnectionError`、日志也打了「已降级」，但 `router.backend` 仍然是 `redis`，
  后续读操作继续打向已故障的后端 —— 而 `memory` 里其实已经有数据了。
- 原因：降级条件写成了 `if self._active is self._redis: self._active = self._memory`。
  也就是「**只有当 active 恰好是那个 Redis 实例时才降级**」。生产路径上二者通常同一对象，
  所以平时看不出问题；但只要 active 指向别的实例（单测注入、将来重构、
  降级与恢复的中间态），这条判据就不成立，降级变成静默失效。
- 解决：判据改为「**当前用的不是 memory 就换到 memory**」`if self._active is not self._memory`，
  语义变成目标而不是身份，天然覆盖所有来源。
- 教训：
  1. **降级判据不要用对象身份**（`is` 比较具体实例），要用目标状态（"是不是兜底实现"）；
     身份比较在正常路径上等价，出问题时却完全失效；
  2. 判据类代码要有**独立断言**：`assert router.backend == "memory"`，
     不能只看"日志打了降级"或"结果没抛异常" —— 这两者在 bug 存在时依然成立；
  3. 排查时先打印中间状态（backend / 兜底存储内容）再改代码，
     这次就是靠 `print(router.backend, router._memory._state)` 一眼定位的。
- 标签：降级,判据,对象身份,测试定位

### 14. redis-py 装新版就连不上 Redis 5.x：HELLO / RESP3 握手

- 日期：2026-10-16
- 现象：本机装了 Redis 5.0.14（Windows 版）做真机验证，`redis-cli ping` 通 ✓、
  服务端日志也说 "Ready to accept connections" ✓，但 Python 侧一连接就
  `ResponseError: unknown command 'HELLO', with args beginning with '3'` ✗。
  而 `pip install "redis>=5.0"` 装回来的是 **8.1.0** ✗。
- 原因：redis-py 5.0 起在建立连接时会发 `HELLO` 握手切 **RESP3**，
  而 `HELLO` 是 **Redis 6.0** 才有的命令。5.x 服务端不认 → 直接报错 ✗。
  表现极具误导性：服务端完全正常、命令行客户端一切正常，只有 Python 连不上，
  很容易误判成"服务端挂了"或"网络问题"，实际是客户端与服务端协议版本不匹配。
- 解决：`Redis.from_url(..., protocol=2)` 显式锁 RESP2（本项目用不到 RESP3 的任何特性 ✓），
  这样与 redis-py 版本解耦：无论装 5.x 还是 8.x，连接 5.x/6.x/7.x 服务端都一致 ✓。
- 判据：同一个脚本在 redis-py 8.1.0 + Redis 5.0.14 组合下 16 项全通过 ✓。
- 附带（同一轮踩到）：Redis 的 `dir ./` 是**相对路径**，若当前工作目录不可写，
  后台 RDB 保存会失败，而 `stop-writes-on-bgsave-error` 会把**所有写命令**全禁掉
  （报 `MISCONF Redis is configured to save RDB snapshots...`）。
  这个现象对应用层来说就是"Redis 看起来在跑但一律写失败"——
  好在降级链接住了（写失败→回退进程内+告警），但排查时必须先想到这一层，
  否则会误以为是自己代码的问题。纯本地验证用 `--save ""` 关掉 RDB 最省事。
- 标签：redis-py,RESP3,协议版本,MISCONF,真机验证

### 15. 替身和真实对象一起自洽，测试全绿而真机 /health 直接 500

- 日期：2026-10-16
- 现象：起两个真实进程接上真 Redis 验收跨副本取消，第一个 `/health` 就 **HTTP 500**：
  `AttributeError: 'TaskStoreRouter' object has no attribute 'subscribe_cancel'` ✗。
  而同一份代码的 31 + 30 条单测**全绿** ✗。
- 原因：`tasks.py` 是在**路由器**（`TaskStoreRouter`）上发起订阅的，
  而我把 `subscribe_cancel` 只加在了抽象基类与两个具体后端上 ✓，**唯独漏了路由器** ✗。
  单测没抓到，是因为那条用例里我自己写了个 `_FakeRouter` 替身 ✓，
  替身**自带** `subscribe_cancel` ✓ —— 于是「替身」与「我以为的接口」完全吻合 ✓，
  真实对象的缺陷被完美遮住 ✓。更糟的是这类替身会让测试**看起来更可信**（它"验证"了订阅只注册一次 ✓）。
- 解决：① 给路由器补上转发方法；② **把那条用例改成用真实的 `TaskStoreRouter`**
  （只把后端换成假 Redis 客户端 ✓），让缺方法必然暴露；
  ③ 另加一条纯回归用例直接断言 `callable(TaskStoreRouter().subscribe_cancel)` ✓。
- 同轮真机还抓到第二处：跨副本取消时 `/chat/cancel` 返回
  `{"cancelled": true, "status": "not_found"}` —— 取消**已经生效**（共享状态里取消原因写进去了 ✓），
  状态字段却还报「没找到任务」 ✗。原因是 `status` 只读本地句柄 ✓，
  任务在别的副本上时必然 None ✓。改为本地没有就读共享状态 ✓，
  并加回归用例钉住 ✓。这类"**行为对了但自述错了**"的缺陷单测基本抓不到 ✓，
  因为它不违反任何断言 ✓，只违反调用方的预期 ✓。
- 教训：
  1. **替身必须尽量用真实对象**，只替换最外层的 IO 依赖（这里换成假 Redis 客户端）；
     自己写替身时，至少要有一条例外用例断言"真实类型也具备这些方法" ✓；
  2. **接口的每个转发层都要有真实对象的回归用例** —— 抽象基类有方法 ≠ 路由器有方法 ✓；
  3. 真机验证不能省 ✓：本轮三个缺陷（HELLO 握手、路由器漏方法、status 自述矛盾）**全部**是
     装上 Redis 跑真进程才暴露的 ✓，其中两个在 500+ 条单测全绿的情况下活着。
- 标签：替身陷阱,真机验证,回归用例,接口转发

### 16. 加了 `.env.prod` 却忘了加进 .gitignore —— 密钥只差一次 `git add` 就泄露

- 日期：2026-10-16
- 现象：为上云准备时新增了生产密钥文件 `.env.prod`（模型 Key + `API_KEY` + CORS 白名单）✓，
  写完顺手用 `git check-ignore` 验了一下 —— **没有被忽略** ✗。
  也就是说任何人（或任何一条 `git add .`）都会把明文密钥提交进仓库 ✓，
  而密钥一旦进了 git 历史，事后删文件是不够的（历史里还在 ✓）。
- 原因：`.gitignore` 里只有 `.env`，是当初按"本地开发配置文件"加的 ✓；
  新增 `.env.prod` 时没有同步这条规则 ✓ —— 这是**改一处忘一处**的典型 ✗。
- 解决：`.gitignore` 补 `.env.prod` 与 `.deploy-state`（部署状态文件）✓，
  并在 `docs/DEPLOY.md` 与 README 里都写明"密钥只落服务器本地文件" ✓。
- 预防（比修更重要）：
  1. **新增任何可能含密钥的文件名，必须当场 `git check-ignore` 验一次** ✓
     —— 比读 .gitignore 可靠，因为它验的是 git 的实际行为 ✓；
  2. 部署检查清单里加一条「`git status` 里不能出现 `.env*`」✓；
  3. 仓库 Settings 里开 **push protection / secret scanning**（本次未配置 ✓，属待办 ✓）。
- 顺带（同轮检查出来但属设计取舍、已在文档写明）：`/health` 会真实调一次 LLM + Embedding，
  拿它做 compose/nginx 的健康检查会**每 30 秒烧一次 token**，且模型超时会判定容器不健康
  → 被反复重启，越探测越坏。已新增零外部依赖的 `/healthz` 专供探活 ✓。
- 标签:密钥泄露,gitignore,上云,健康检查

### 17. 测试被本机 `.env` 污染：本地全红、CI 全绿

- 日期：2026-10-16
- 现象：给 `.env` 打开 `TASK_STATE_BACKEND=redis`（上云前启用 Redis）之后，
  本地 `mvn test` 等价物里 **8 条用例红**（4 条 store + 4 条 tasks ✓），
  而同样的代码在 CI 上**全绿** ✓。
- 原因：那 8 条用例隐式依赖「默认后端是进程内」✗，
  并没有在测试里写出这个前提 ✓。`.env` 被 gitignore ✓ → CI 上没有这个文件 ✓ →
  `config.TASK_STATE_BACKEND` 取代码默认值 `memory` ✓ → 测试通过 ✓；
  本机有 `.env` → 读到 `redis` → 测试红 ✓。
  换句话说，**测试的成败取决于开发者本机的配置文件** ✗ —— 这是最坏的一类脆弱性 ✓。
- 解决：两个文件的 autouse fixture 都显式 `monkeypatch.setattr(config, "TASK_STATE_BACKEND", "memory")` ✓，
  让"默认应为进程内"这个前提由测试自己声明 ✓。
- 教训：
  1. **凡是"默认应为 X"的测试，必须把 X 写出来** ✓ —— 不能依赖 config 的默认值 ✓；
  2. `.env` 被 gitignore 是对的（防密钥泄露 ✓），但代价是**本地与 CI 配置必然不同** ✓，
     所以测试必须对配置差异免疫 ✓，而不是"我本地一直是这样的" ✓；
  3. 排查"本地红 CI 绿"时，先查**本地配置文件**（`.env` / `local-settings.xml` ✓），
     别急着怀疑代码或版本不一致 ✓ —— 这次第一反应确实是"代码写错了"，方向错了 ✓。
- 标签:测试隔离,本地配置,.env,CI一致性


### 18. 压测脚本连踩四坑，其中两个会让「数字看起来正常、实际全是错的」

- 日期：2026-10-16
- 背景：补 locust 阶梯压测（`bench/run_bench.ps1` + `bench/locustfile.py` + `bench/plot_bench.py`）。
- 四个坑：
  1. **不加 `--headless` 会挂死** —— locust 会尝试起 Web UI 并阻塞，脚本一直不返回，
     最后被外部取消。第一次跑 20 秒的场景花了 4 分多钟才发现卡住。
  2. **`--tags` / `--class-picker` 在 locust 2.46 上都会把类过滤成空** ——
     `--class-picker` 写类名或正则都报 `Unknown User(s)`；
     `--tags 'baseline or retrieval'`、`--tags 'tag:retrieval'` 不报错但**跑出 0 请求**、
     汇总表全空（locust 的 tag 匹配的是 **Task 级 `@tag`**，不是 User 类上的 `tags`）。
     最终方案：用 `BENCH_SCENARIO` 环境变量把其余类的 `weight` 置 0 ——
     `weight = 0` 是唯一被实测确认能排除 User 类的机制。
  3. **`-u 1` 时 locust 只派一个 User 类** —— 按权重随机选，于是「1 并发」那档可能整段只有
     baseline，检索行直接缺失，看起来像「检索在低并发下没有数据」。改为每场景单独一轮。
  4. **画图脚本的 glob 误读了 `_stats_history.csv`** —— 它也有 `Name` 字段，解析后把正确
     统计覆盖成空，症状是「汇总表全空、但 CSV 明明有数据」。
- 另外三个**方法论**问题（比上面的坑更要紧）：
  - **限流必须先关**：第一次跑 195 个请求全是 429 —— `API_RATE_LIMIT=60`/IP/分钟，
    而每档只压 20 秒，全部落在同一个窗口内。顺带得到一个更有价值的结论：
    **默认配置下单 IP 吞吐上限约 1 req/s**，比系统自身的拐点早得多。
  - **必须预热**：实测**首次检索 41.9 秒**（要加载本地 BGE + faiss），第二次起约 100ms。
    不预热就压，测到的是「模型加载有多慢」而不是系统性能。
  - `.env` 用的是 `load_dotenv(override=True)` —— **`.env` 会覆盖进程环境变量**，
    所以 `$env:API_RATE_LIMIT=0` **无效**，必须改 `.env` 再重启。
- 教训：
  1. **「参数不报错 ≠ 参数生效」** —— `--tags` 那次不报错、退出码 0、只是 0 请求；
     验收压测脚本要看**有没有请求**，不能只看退出码；
  2. **中间产物要先看一眼** —— 表格全空时先读那个 CSV，就能立刻分清「没数据」还是「解析错了」；
  3. **压测脚本的第一版往往在错的口径上跑得很顺** —— 限流、冷启动、场景混跑
     都不会报错，只会给你一份「看起来挺合理」的错数据。
- 产出：`docs/BENCHMARK.md`（原始数据、曲线、Little's Law 校验、复现命令与口径声明）。
- 标签:压测,locust,方法论,限流,预热

### 19. 补哈希脚本给「带环境标记的行」多加了一个反斜杠，Linux CI 直接挂

- 日期：2026-10-16
- 现象：合并到 master 后 CI 里 6 个检查过了 5 个，`lock-verify` 红了：
  ```
  ERROR: Invalid requirement: 'uvloop==0.22.1': Expected end of marker expression
      sys_platform != "win32" ^
  ```
  而**本机装得好好的**（Windows 根本不装 uvloop，所以本地永远复现不出来 ✗）。
- 原因：`scripts/gen-lock-hashes.py` 给每个包的第一行追加 `\` 做续行 ✓✓，
  但**带环境标记的依赖，pip-compile 输出的那一行本身就已经以 `\` 结尾** ✓：
  ```
  uvloop==0.22.1 ; sys_platform != "win32" \
  ```
  再补一个就成了 `\ \` ✗✗ —— pip 解析到反斜杠处直接断裂 ✓。
  这与 #4 是**同一个根因的两个变种**：#4 是手写 hash 行漏反斜杠 ✓，
  这次是脚本给已带续行的行**重复加** ✓。
- 修复：
  1. 脚本改成先看该行是否已以 `\` 结尾 ✓✓ —— 幂等 ✓，重跑不会再次叠加 ✓；
  2. 两份锁各修 1 行 ✓（`\ \` → `\` ✓）；
  3. 用 **pip 自己的解析器**验证而不是靠"本机能装"：
     `pip install --dry-run --no-index --no-deps --require-hashes -r <lock>`
     ✓✓ —— `--no-index` 不联网、纯解析 ✓，`Invalid requirement` 消失即证明解析通过 ✓✓。
     这是关键判据：**"本机装得上"不能证明锁文件正确** ✗✗（平台标记的依赖在 Windows 上被跳过 ✓）。
- 教训：
  1. **凡是"给既有行追加内容"的脚本，都必须幂等** ✓ —— 不幂等的脚本第二次运行就会破坏文件 ✓，
     而它平时看起来完全正常 ✓；
  2. **跨平台产物（锁文件）的验证必须在目标平台上做** ✓✓ ——
     `check_lock_platform_deps.py`（Linux 依赖闭包 ✓）+ pip 解析器（语法 ✓）两条一起 ✓✓，
     缺一条就会漏出这类问题 ✓；
  3. 报错信息里那个孤零零的 `^` 是关键线索 ✓ —— 它指向的是**上一行的续行符** ✓，
     不是 `sys_platform != "win32"` 这段标记本身 ✓✓（第一次差点查错方向 ✗）。
- 标签:锁文件,哈希模式,续行,幂等,跨平台,CI