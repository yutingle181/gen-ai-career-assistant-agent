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

