#!/usr/bin/env bash
# 服务器端部署脚本：拉代码 → 重建镜像 → 滚动重启 → 探活。
#
# 由 .github/workflows/deploy.yml 通过 SSH 调用，也可以手工执行（便于排障）：
#   cd /opt/genai-career-assistant && DEPLOY_PATH=$PWD IMAGE_TAG=$(git rev-parse HEAD) bash scripts/deploy/remote_deploy.sh
#
# 设计要点（每条都是为了"部署失败时不把线上搞挂"）：
#   1. `set -euo pipefail`：任何一步失败立刻退出，不带着半成品继续；
#   2. `git pull --ff-only`：只允许快进，不在生产机上生成 merge commit；
#   3. 先 build 再 up -d：build 失败时旧容器还在跑（compose 的 up 根本没执行）；
#   4. 部署前把当前 commit 记进 .deploy-state，回滚就是 checkout 回去再 up -d；
#   5. 探活失败明确报错：宁可让流水线红，也不要"看起来部署成功了"。
#
# 未验证声明（2026-10-16）：本脚本写就绪但**未在真实服务器上执行过**（当时还没有服务器）。
# 首次使用请先在服务器上 dry-run 一次（把 DRY_RUN=1 传进去）。

set -euo pipefail

DEPLOY_PATH="${DEPLOY_PATH:-$(cd "$(dirname "$0")/../.." && pwd)}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "$DEPLOY_PATH" rev-parse HEAD)}"
COMPOSE_FILE="docker-compose.prod.yml"
DRY_RUN="${DRY_RUN:-0}"
cd "$DEPLOY_PATH"

log() { printf '[deploy] %s\n' "$*"; }

if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1，只做检查不改动任何东西"
fi

# ---- 1. 前置检查：缺什么当场说清，不要等到构建失败才报错
command -v docker >/dev/null 2>&1 || { log "未安装 docker"; exit 1; }
docker compose version >/dev/null 2>&1 || { log "docker compose 不可用（需要 Compose V2）"; exit 1; }
[[ -f ".env.prod" ]] || { log "缺少 .env.prod —— 从 .env.example 复制并填入模型 Key / API_KEY / CORS_ORIGINS"; exit 1; }
# 密钥权限：.env.prod 里是模型 Key 与 API_KEY，权限放宽等于把密钥公开
if stat -c '%a' .env.prod 2>/dev/null | grep -qv '^600$'; then
    log "提示：.env.prod 权限不是 600，建议执行 chmod 600 .env.prod"
fi
mkdir -p data/knowledge data/index

# ---- 2. 记录当前版本（回滚要用）
PREV_COMMIT="$(git rev-parse HEAD)"
log "当前 commit = $PREV_COMMIT"
if [[ "$DRY_RUN" == "1" ]]; then
    log "将执行：git pull --ff-only → docker compose build → up -d → 探活"
    git fetch --all --prune
    git log --oneline HEAD..origin/master || true
    docker compose -f "$COMPOSE_FILE" config --quiet && log "compose 文件语法通过"
    exit 0
fi

# ---- 3. 拉代码（只快进）
git fetch --all --prune
git checkout master
git pull --ff-only origin master
NEW_COMMIT="$(git rev-parse HEAD)"
log "更新到 $NEW_COMMIT"

# 镜像 tag 用 commit sha：出问题能一眼看出线上跑的是哪次构建
log "构建镜像（tag=$IMAGE_TAG）"
IMAGE_TAG="$IMAGE_TAG" docker compose -f "$COMPOSE_FILE" build

# ---- 4. 重建容器
# 注意顺序：先 up -d 再探活。build 失败的话旧容器还在跑（上面的 set -e 已经拦住）。
IMAGE_TAG="$IMAGE_TAG" docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

# ---- 5. 探活（零外部依赖的 /healthz）
log "等待服务就绪…"
for i in $(seq 1 30); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
        printf '%s\n' "$PREV_COMMIT" > .deploy-state
        log "部署成功：$(curl -fsS http://127.0.0.1:8000/healthz)"
        docker compose -f "$COMPOSE_FILE" ps
        log "回滚方式：git checkout ${PREV_COMMIT} && IMAGE_TAG=${PREV_COMMIT} bash scripts/deploy/remote_deploy.sh"
        exit 0
    fi
    sleep 2
done

log "部署后探活失败（60 秒内 /healthz 没通），当前状态："
docker compose -f "$COMPOSE_FILE" ps || true
docker compose -f "$COMPOSE_FILE" logs --tail=80 api || true
exit 1