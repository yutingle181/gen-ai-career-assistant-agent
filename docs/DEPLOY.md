> **验证状态（2026-10-16）**：本文档与配套的 compose / 脚本 / CI 在**没有服务器**的前提下写就绪，
> 静态检查已过（compose 语法、ruff、518+ 条单测），但**未在真实服务器上执行过一次**。
> 首次部署请按下面的「验收清单」逐项确认，不要假设它开箱即用。
> 部署过程中踩到的坑请追加到 `DEV_NOTES.md`。

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