# Agent实习项目 开发记录（DEV_NOTES）

> 本文件由 `preflight-check` skill 维护：动手前先读本文件，踩坑后追加记录。
> 生成时间：2026-09-19 23:24
> 重新生成：`dev-notes.ps1 -Init -Force`；追加记录：`dev-notes.ps1 -Add`

## 一、项目速览

- 项目名称：Agent实习项目
- 版本号：1.5.0
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

