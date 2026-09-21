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
- 判据：在 Linux 上 `pip install --require-hashes -r requirements-dev.lock.txt` 成功（CI 的 lock-verify 已覆盖）。
  **注意这类问题在 Windows 上无法复现**——「平台专属依赖是否齐全」没有便宜的静态判据，只能真在 Linux 装一次，
  这正是 lock-verify 与镜像构建的价值所在。
- 标签：pip,锁文件,跨平台,哈希模式,uvloop,供应链

