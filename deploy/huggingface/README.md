---
# Hugging Face Spaces 元数据（sdk: docker = 用我们的 Dockerfile 直接构建）
# 这份 README 推到 HF 后位于仓库**根目录**，所以只放平台需要的信息 ✓；
# 项目文档仍是仓库里的 README.md（两者不冲突 ✓✓）。
title: GenAI Career Assistant
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# GenAI Career Assistant（AI 求职助手）

LangGraph 多 Agent + 企业知识库 RAG 的求职辅助系统。**这个 Space 是 API 服务** ✓✓
（Web 界面在另一个仓库 ✓）。

## 能用什么

| 端点 | 说明 |
| --- | --- |
| `GET /healthz` | **零外部依赖探活** ✓✓（不调模型 ✓✓ 适合平台健康检查 ✓✓） |
| `GET /health` | 深度诊断（**会真实调一次 LLM + Embedding** ✓✗ 手动点开看 ✓✓） |
| `POST /chat` | 同步对话 |
| `POST /chat/stream` | SSE 流式对话 |
| `POST /chat/cancel` | 取消在途任务 |
| `GET /knowledge` · `POST /knowledge/{kb}/ingest` · `POST /knowledge/{kb}/search` | 知识库 RAG |

## 已知限制（部署方必须知道 ✗✗）

1. **48 小时无访问会 sleep** ✗✗ —— 首次访问要等 **30–60 秒**冷启动 ✓✓
   （HF 免费档行为 ✓✓，无法关闭 ✗ —— 面试当天**提前几分钟点开** `/healthz` ✓✓）
2. **容器文件系统重启后清空** ✗✗（除挂载的 Dataset ✓✓）
   → 知识库索引每次冷启动后需重建（`POST /knowledge/{kb}/ingest_dir` ✓✓），
   `samples` 库只 3 个文档 ✓✓ 约 10 秒 ✓✓
3. **单容器** ✗✗ —— 这里验证的是**镜像本身** ✓✓，
   多服务编排（api + redis + 数据卷）请参考仓库里的 `docker-compose.prod.yml` ✓✓

## 密钥怎么配

**不在镜像里** ✓✓（镜像已排除 `.env` ✓✓）—— 在 Space 的
`Settings → Variables and secrets` 里加 ✓✓：

| Secret 名 | 是否必填 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | **必填** ✗✗ | 缺了 `/chat` 全部降级 ✓✗ |
| `OPENAI_BASE_URL` | 必填 | 默认 `https://api.deepseek.com` |
| `MODEL_NAME` | 必填 | 默认 `deepseek-chat` |
| `EMBEDDING_PROVIDER` | 选填 | `local` = 本地 BGE（**免 Key** ✓✓ 本 Space 默认 ✓✓） |
| `LOCAL_EMBEDDING_MODEL` | 选填 | 默认 `BAAI/bge-small-zh-v1.5` |
| `API_KEY` | **强烈建议** ✓✓ | 给业务接口加鉴权，否则谁都能烧你的额度 ✗✗ |
| `PORT` | 平台自动设 ✓ | 7860，**不用手动加** ✓✓ |

> ⚠️ `EMBEDDING_PROVIDER=local` 时不需要 Embedding Key ✓✓
> 但首次要下载 ~100MB 模型 ✓✓ —— 冷启动会慢一些 ✓✓（已缓存后正常 ✓✓）

## 技术栈

FastAPI · LangGraph · BM25 + 向量混合检索 · RRF 融合 · Rerank · diskcache · SSE 流式 · 可取消任务

## 源码

完整源码、测试、压测与部署文档见 GitHub 仓库 ✓✓
（这个 Space 只是它的部署产物之一 ✓✓）。