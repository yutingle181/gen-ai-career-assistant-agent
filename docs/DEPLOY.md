> **验证状态（2026-10-16）**：本文档与配套的 compose / 脚本 / CI 在**没有服务器**的前提下写就绪，
> 静态检查已过（compose 语法、ruff、518+ 条单测），但**未在真实服务器上执行过一次**。
> 首次部署请按下面的「验收清单」逐项确认，不要假设它开箱即用。
> 部署过程中踩到的坑请追加到 `DEV_NOTES.md`。
>
> **2026-10-17 补充**：新增 §12「零成本真机验证（GitHub Codespaces + cloudflared）」——
> 不想买服务器也能真机部署 ✓✓；§11 的 Hugging Face Spaces 经查证**需要 PRO 订阅** ✗✗
> （免费档只能建 Static Space ✓✗），一并写明以免误导 ✓。

# 部署指南（生产）

## 0. 方案与取舍

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 镜像怎么送 | **服务器上 `git pull` + `docker compose build`** | 零凭据、零额外账号（不注册 Docker Hub、不开 ACR）；回滚就是 `git checkout` 回旧 commit。代价是每次部署重建镜像（几分钟）—— 单人项目完全可接受 |
| 镜像 tag | commit sha | 线上跑的是哪次构建一眼可查；回滚有确定目标 |
| 对外暴露 | 容器只绑 `127.0.0.1`，对外由 nginx / 云安全组决定 | 避免绕过反代与 TLS；`docker-compose.yml`（开发版）的 `8000:8000` 是全网卡，服务器上等于裸奔 |
| 探活 | `/healthz` | `/health` 会**真调一次 LLM + Embedding**：每 30 秒烧 token、模型超时会判定容器不健康导致反复重启（越探测越坏）。`/healthz` 零外部依赖 |
| Redis | 生产 compose 自带，任务状态跨副本共享 | 不配也能跑（自动降级为进程内并告警），多副本时必须配，否则 cancel 打到另一个副本会静默失败 |

## 1. 服务器前置

按 `IP + 端口` 起步（国内服务器想用 80/443 绑域名，**必须先备案**；未备案只能用非标准端口）。

```bash
# 系统与 Docker（以 Ubuntu 22.04/24.04 为例）
sudo apt update && sudo apt install -y git curl ca-certificates
# Docker Engine + Compose V2（官方源，别用 snap 的旧版）
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 免密 sudo（仅限 docker 这一条，别给全量 NOPASSWD）
sudo usermod -aG docker $USER && newgrp docker
docker compose version   # 应输出 Compose V2
```

安全组：放行 `22`（SSH，**只对自己的 IP 放行**）、`8000`/`8501`（先用非标准端口跑通；备案后再上 80/443）。

## 2. 首次部署

```bash
sudo mkdir -p /opt/genai-career-assistant && sudo chown -R $USER /opt/genai-career-assistant
git clone https://github.com/yutingle181/gen-ai-career-assistant-agent.git /opt/genai-career-assistant
cd /opt/genai-career-assistant

cp .env.example .env.prod
vi .env.prod      # 至少填这些，见下节
chmod 600 .env.prod

mkdir -p data/knowledge data/index Agent_output
# 知识库文档放进 data/knowledge/<kb>/（镜像里**不含** data/，必须挂进来）

docker compose -f docker-compose.prod.yml config --quiet   # 先验语法
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
curl -fsS http://127.0.0.1:8000/healthz && echo
curl -fsS http://127.0.0.1:8000/health | head -c 400        # 看鉴权/限流/CORS 风险项
```

## 3. `.env.prod` 必填项

| 键 | 为什么必填 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` | 缺了服务能起，但 `/chat` 全部降级，等于不可用 |
| `EMBEDDING_API_KEY` 或 `EMBEDDING_PROVIDER=local` | 建库必需；local 模式不需 Key，但要 `pip install -r requirements-optional.txt` 并重建镜像 |
| `API_KEY` | **公网必填**：留空 = 任何人可调用 `/chat` 消耗你的额度 |
| `CORS_ORIGINS` | 填你的前端来源；`*` 与 `allow_credentials` 是非法组合，浏览器会直接拒 |
| `API_RATE_LIMIT` | 保持开启（默认 60/分钟/IP） |
| `TASK_STATE_BACKEND=redis` | 多副本时必填，否则取消请求打到别的副本会静默失败 |

启动后先看 `/health` 的 `runtime.public_exposure_risks`，它会明确列出还差哪几项安全配置。

## 4. 首次部署后建知识库索引

镜像不含 `data/`，所以第一次部署必须建索引（否则 RAG 是空的、`/knowledge` 返回空列表）：

```bash
# 方式 A：接口建库（推荐，与线上走同一条路径）
curl -fsS -X POST http://127.0.0.1:8000/knowledge/samples/ingest_dir \
     -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
     -d '{"directory": "data/knowledge/samples"}'

# 方式 B：CLI（离线，更适合大文件）
docker compose -f docker-compose.prod.yml exec api \
     python run_eval.py --kb samples --rebuild
```

换向量模型或维度后**必须重建索引**（faiss 维度与向量模型绑定，维度不匹配会直接检索失败）。

## 5. nginx 反代（可选，但上公网建议做）

用仓库里的模板：`deploy/nginx.conf.example`，把 `SERVER_NAME` 与证书路径改掉。

```bash
sudo cp deploy/nginx.conf.example /etc/nginx/conf.d/genai.conf
sudo vi /etc/nginx/conf.d/genai.conf
sudo nginx -t && sudo systemctl reload nginx
```

未备案时只能用 `<IP>:8000` 直连 + 明文 HTTP；备案后再上域名与 Let's Encrypt 证书。

## 6. 日常更新（CI 自动）

推 `master` 且改动命中运行文件（`src/**`、`Dockerfile`、`requirements*.txt`、`docker-compose.prod.yml`、`scripts/deploy/**`）时，`.github/workflows/deploy.yml` 会：

1. SSH 到服务器执行 `scripts/deploy/remote_deploy.sh`；
2. 服务器 `git pull --ff-only` → `docker compose build` → `up -d`；
3. 探活 `/healthz`，失败则流水线红。

**需要配置的 Secrets**（仓库 Settings → Secrets）：

| Secret | 说明 |
| --- | --- |
| `DEPLOY_HOST` | 服务器 IP |
| `DEPLOY_USER` | 部署用户（需在 docker 组内） |
| `DEPLOY_SSH_KEY` | 私钥全文（不是 `.pub`） |
| `DEPLOY_PATH` | 仓库路径，如 `/opt/genai-career-assistant` |

## 7. 回滚

```bash
cd /opt/genai-career-assistant
PREV=$(cat .deploy-state)              # 上一次成功部署的 commit
git checkout "$PREV"
IMAGE_TAG="$PREV" docker compose -f docker-compose.prod.yml up -d --build
```

回滚同样要先 `build` 成功再 `up -d`，否则旧镜像被覆盖且新镜像起不来。`build` 失败时旧容器仍在跑（`set -e` 会在 build 后停住）。

## 8. 备份与恢复

需要备份的三样：`.env.prod`（密钥）、`data/`（知识库 + 索引 + `app.db`）、`data/checkpoints.sqlite`（图 checkpoint）。

```bash
# 备份（加密后传走；别把未加密的密钥文件留在机器上）
tar czf /tmp/genai-backup-$(date +%F).tar.gz .env.prod data/
# 恢复
tar xzf /tmp/genai-backup-2026-10-16.tar.gz -C /opt/genai-career-assistant
docker compose -f docker-compose.prod.yml up -d
```

## 9. 排障速查

| 现象 | 大概率原因 |
| --- | --- |
| 容器起不来 / 一直 restarting | `docker compose logs --tail=100 api`；多半是 `.env.prod` 缺模型 Key 或端口被占 |
| `/healthz` 通但 `/chat` 报错 | 看 `/health` 的 `llm_detail` / `embedding_detail`：多半是 Key 额度耗尽（`Free quota exhausted`），不是部署问题 |
| `/knowledge` 返回空 | 镜像不含 `data/`，第 4 节没建索引 |
| 检索报维度错误 | 换过向量模型没重建索引 → 第 4 节 `--rebuild` |
| 取消无效（多副本） | `TASK_STATE_BACKEND` 没设 redis，或 Redis 起不来（看 `/health.runtime.task_state_backend`） |
| Redis 写全失败 `MISCONF` | 容器内 `dir` 不可写导致 RDB 保存失败 → 生产 compose 已用 `--save ""` 规避 |
| Streamlit 页面白屏 | 反代未透传 `X-Forwarded-*`；先直连 `:8501` 判断是不是反代问题 |
| 内存高 | Redis 已限 256mb；应用侧可给 uvicorn 加 `--workers`，但**多 worker 需要 Redis 已启用**（否则取消失效） |

## 10. 首次部署验收清单

- [ ] `curl /healthz` 返回 200
- [ ] `curl /health` 的 `runtime.public_exposure_risks` **为空数组**
- [ ] `curl /health` 的 `runtime.task_state_backend` 为 `redis`
- [ ] 不带 `X-API-Key` 调 `/chat` 返回 401
- [ ] `/knowledge` 能列出知识库，且 `/chat` 能引用到文档内容
- [ ] 容器内 `id -u` 为 10001（non-root）
- [ ] `docker compose restart api` 后 SSE 能正常收尾（无残留进程）
- [ ] 故意写错一次 `.env.prod` 里的模型 Key → 确认容器**不会**陷入重启循环（探活不该被模型拖垮）

---

## 11. Hugging Face Spaces（**需要 PRO 订阅，非免费**）

> ⚠️ **2026-10-17 查证后更正**：本节原计划是「零成本上云」，但 HF 已改政策 ——
> **免费档只能建 Static Space** ✓✗，**Gradio 与 Docker Space 都要 PRO（$9/月）** ✗✗。
> 官方定价页把 "Host/Create Gradio & Docker Spaces" 明确列为 PRO / Team 能力 ✓。
>
> 所以：**不想付费就别走这条路** ✓✓ —— 改用 §12 的 Codespaces 演练 + cloudflared 临时链接 ✓✓
> （两者都不花钱 ✓✓ 且 Codespaces 能完整验证 Docker 部署 ✓✓）。
>
> 本节保留是因为：① 已有 HF 账号的话升级 PRO 仍是最省事的公网部署 ✓✓；
> ② `deploy/huggingface/` 下的文件与 `Dockerfile` 的 `PORT` 可覆盖设计**本身仍有用** ✓✓
> （换任何要求固定端口的平台都用得上 ✓✓）。

### 与自建服务器的差别（先看清再决定 ✓✓）

| 维度 | 自建服务器 | HF Spaces（需 PRO ✓✗） |
| --- | --- | --- |
| 费用 | 按月付费 | **$9/月起** ✗✗ |
| 多服务编排（api + redis + 数据卷） | ✅ `docker-compose.prod.yml` | ❌ **单容器** ✓✗ |
| 数据持久化 | ✅ 命名卷 | ❌ **重启后文件系统清空** ✗✗ |
| 常驻 | ✅ | ⚠️ **48h 无访问会 sleep** ✗✗ |
| 端口 | 自定 | 平台约定 **7860** ✓ |
| 密钥 | 服务器本地 `.env.prod` | Space 的 Secrets ✓✓ |

### 与自建服务器的差别（先看清再决定 ✓✓）

| 维度 | 自建服务器 | HF Spaces |
| --- | --- | --- |
| 费用 | 按月付费 | **$9/月起（PRO）** ✗✗ |
| 多服务编排（api + redis + 数据卷） | ✅ `docker-compose.prod.yml` | ❌ **单容器** ✓✗ |
| 数据持久化 | ✅ 命名卷 | ❌ **重启后文件系统清空** ✗✗ |
| 常驻 | ✅ | ⚠️ **48h 无访问会 sleep** ✗✗ |
| 端口 | 自定 | 平台约定 **7860** ✓ |
| 密钥 | 服务器本地 `.env.prod` | Space 的 Secrets ✓✓ |

### 三个必须知道的行为 ✗✗

1. **48 小时无访问会 sleep** ✗✗ —— 首次访问等 **30–60 秒**冷启动 ✓✓。
   **面试/演示当天提前几分钟点开 `/healthz`** ✓✓ 别现场干等 ✗✗。
2. **容器文件系统重启后清空** ✗✗（除挂载的 Dataset ✓✓）
   → 知识库索引每次冷启动后要重建 ✓✓；`samples` 库只 3 个文档 ✓✓ 约 10 秒 ✓✓ 可接受 ✓✓。
3. **单容器** ✓✗ —— 这里验证的是**镜像本身** ✓✓；
   多服务编排仍按 §1–§10 在服务器上做 ✓✓（两者互补 ✓✓ 不是替代 ✗✗）。

### 部署步骤

```powershell
# 1. 先 DryRun，确认导出内容（会打印文件清单与总大小 ✓✓）
powershell -ExecutionPolicy Bypass -File deploy\huggingface\push-to-hf.ps1 `
    -HfRemote <你的HF用户名>/<space名> -DryRun

# 2. 确认无误后真正推送（HF 会自动 build ✓✓）
powershell -ExecutionPolicy Bypass -File deploy\huggingface\push-to-hf.ps1 `
    -HfRemote <你的HF用户名>/<space名>
```

脚本做三件事 ✓：① 用 `git archive` 导出**已提交**内容 ✓✓（天然排除 `.venv` / `__pycache__` / 本地数据 ✓✓）；
② 用 `deploy/huggingface/README.md` 覆盖根目录那份（HF 要求根目录有元数据 ✓✓）；
③ 推到 HF remote ✓✓。

### 配密钥（Space 页面 → Settings → Variables and secrets ✓）

| Secret | 必填 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | **必填** ✗✗ | 缺了 `/chat` 全部降级 |
| `OPENAI_BASE_URL` | 必填 | 默认 `https://api.deepseek.com` |
| `MODEL_NAME` | 必填 | 默认 `deepseek-chat` |
| `EMBEDDING_PROVIDER` | 选填 | `local` = 本地 BGE（**免 Key** ✓✓ 默认 ✓✓） |
| `API_KEY` | **强烈建议** ✓✓ | 否则谁都能烧你的额度 ✗✗ |
| `PORT` | 平台自动设 ✓ | 7860，**不用手动加** ✓✓ |

> ⚠️ `EMBEDDING_PROVIDER=local` 时首次要下载 ~100MB 模型 ✓✓ 冷启动会慢一些 ✓✓。

### 部署后验收

```powershell
# 1. Build 日志全绿（Space 页面 → Build 页 ✓）
# 2. 先点开触发冷启动（别等面试时干等 ✗✗）
curl https://<用户名>-<space名>.hf.space/healthz
# 3. 深度诊断（会真调一次模型 ✓✗ 手动点开看 ✓✓）
curl https://<用户名>-<space名>.hf.space/health
# 4. 建知识库索引（冷启动后文件系统是空的 ✗✗）
curl -X POST https://<用户名>-<space名>.hf.space/knowledge/samples/ingest_dir `
     -H "X-API-Key: <你设的Key>" -H 'Content-Type: application/json' `
     -d '{"directory":"data/knowledge/samples"}'
```

### 端口是怎么做到 8000/7860 共存的 ✓

`Dockerfile` 里是 `ENV PORT=8000` + `CMD ["sh","-c","exec ... --port ${PORT}"]` ✓✓：

- 本机 / compose / 自建服务器 → 默认 **8000** ✓✓ 行为零变化 ✓✓
- HF Spaces → 平台注入 `PORT=7860` ✓✓ **不用改 Dockerfile** ✓✓
- 用 `sh -c` + `exec` 而不是 exec 数组：数组形式不展开 `${PORT}` ✗；
  `exec` 让 uvicorn 替换掉 sh 成为 PID 1 ✓✓ **SIGTERM 仍能直达** ✓✓（否则优雅退出失效 ✗✗）

> 同样的原因，`HEALTHCHECK` 用的是 `CMD-SHELL` 而不是 exec 形式 ——
> exec 形式不展开环境变量，硬编码 8000 会让 HF 上的探活永远失败 ✗✗。

---

## 12. 零成本真机验证（GitHub Codespaces + cloudflared）

不想买服务器、也不想付 HF PRO ✓✓ —— 这条路**一分钱不花** ✓✓ 且能完整验证 Docker 部署 ✓✓。

### 组合方式（两步互补 ✓✓）

| 步骤 | 工具 | 验证什么 | 费用 |
| --- | --- | --- | --- |
| **A. 部署演练** | **GitHub Codespaces**（免费 **60 小时/月** ✓✓） | `Dockerfile` + `docker-compose.prod.yml` **真的能 build、能起、能探活** ✓✓ | 免费 ✓✓ |
| **B. 临时公网 demo** | **cloudflared 穿透本机** | 给面试官一个**能点开的链接** ✓✓ | 免费 ✓✓ |

> 为什么两个都要：Codespaces 能验证「**我们的部署产物是好的**」✓✓ 但它**不能常驻** ✓✗（停掉就没了 ✓✗）；
> 本机穿透能常驻 ✓✓ 但验证的是「**我这台机器**」而不是「镜像」✓✗。合起来才闭环 ✓✓。

### A. Codespaces 演练（推荐先做这个 ✓✓）

```bash
# 1. GitHub 仓库页面 → Code → Codespaces → Create codespace on <分支>
#    （2 核 / 4GB / 约 30 秒启动 ✓✓）
# 2. 进环境后直接跑（Codespaces 自带 Docker ✓✓）
docker compose -f docker-compose.prod.yml up -d --build

# 3. 逐条验收（与 §10 清单一致 ✓✓）
curl -fsS http://127.0.0.1:8000/healthz && echo
curl -fsS http://127.0.0.1:8000/health | head -c 400

# 4. 验证完**停止** Codespaces ✓✓ —— 不停止会持续消耗免费额度 ✗✗
```

**验收要点** ✓✓（这些是 D 计划至今没验证过的部分 ✓✓）：

- [ ] 镜像能 build 成功（哈希模式装依赖 ✓✓ 这步最容易暴露锁文件问题 ✗✗）
- [ ] 容器内 `id -u` = **10001**（non-root ✓✓）
- [ ] `/healthz` 返回 200 且**不调模型** ✓✓（看耗时 < 200ms ✓✓）
- [ ] `docker compose restart api` 后服务自愈 ✓✓
- [ ] Redis 容器起来后 `/health.runtime.task_state_backend` = `redis` ✓✓
- [ ] 故意把 `.env.prod` 的模型 Key 改错 → 确认**不会**陷入重启循环 ✓✓

> ⚠️ Codespaces 免费额度 **60 小时/月** ✓✓ —— 一次演练约 **1–1.5 小时** ✓✓
> 够演练 40+ 次 ✓✓。**用完即停** ✓✓ 别让它空跑 ✗✗。

### B. cloudflared 临时公网链接（面试当天用 ✓✓）

```powershell
# 1. 装 cloudflared（一次性 ✓✓）
winget install --id Cloudflare.cloudflared

# 2. 本机服务起好后（Agent :8000 已在跑 ✓✓），一条命令出公网链接 ✓✓
cloudflared tunnel --url http://127.0.0.1:8000
```
输出里会有一个 `https://<随机名>.trycloudflare.com` ✓✓ —— **直接发给面试官** ✓✓。

**特点与限制** ✓✓：

| 项 | 说明 |
| --- | --- |
| 费用 | 免费 ✓✓ 无需注册 ✓✓ |
| 链接有效期 | **进程断开即失效** ✗✗（面试时开着就行 ✓✓） |
| HTTPS | 自动 ✓✓ |
| 速度 | 取决于你家宽带上行 ✓✓（聊天场景够用 ✓✓） |
| 安全 | ⚠️ **链接等于公开你的服务** ✗✗ —— 务必先设 `API_KEY` ✓✓ 否则谁都能烧你额度 ✗✗ |

> 💡 想要固定域名 + 常驻 ✓✓ → 那还是得买服务器 ✓✓（见 §1–§10 ✓✓）或付 HF PRO ✓✗。

### 为什么不用其他免费档 ✗✗

| 方案 | 为什么不用 |
| --- | --- |
| ~~HF Spaces~~ | Docker Space 要 PRO（$9/月 ✗✗），免费档只剩 Static ✓✗ |
| ~~Render~~ | 免费实例 **15 分钟不活动就 sleep** ✗✗ 且冷启动要 50s+ ✗ |
| ~~Railway~~ | 只剩 $5 试用额度 ✗✗ 用完要付费 ✗ |
| ~~Fly.io~~ | 2024 年起**取消免费档** ✗✗ |
| ~~GCP Cloud Shell~~ | 会话 **60 分钟不活动即断** ✗✗ 不能常驻 ✓✗ |
| **Oracle Always Free** | 4 核 24GB **永久免费**很香 ✓✓ 但注册要信用卡验证 ✗✗ 国内常卡人机验证 ✗✗ 时间成本太高 ✓✗ |