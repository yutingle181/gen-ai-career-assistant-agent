# RAG 效果评测报告

- 生成时间：2026-10-15 21:17:06
- 模型：`deepseek-v4.1-flash` @ `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Embedding：`local/BAAI/bge-small-zh-v1.5`
- 知识库：`samples`（7 片段 / 3 份文档 / 维度 512）
- 资料新鲜度：过期 0 / 时间未知 0 / 新鲜 7（共 7 个片段，阈值 365 天）
- 评测样本：20 条

## 1. 指标口径

- **Recall@5**：正确片段是否出现在前 5 条召回中（本评测集每条样本 1 个正确片段）
- **MRR**：正确片段首次命中排名倒数的均值，未命中记 0
- **Hit Rate@5**：前 5 条至少命中一条的样本占比
- **幻觉率**：LLM-as-judge 将回答拆为断言，统计不可由召回片段支撑的断言占比（temperature=0，可复现）
- **平均延迟**：单次检索 + 重排的端到端耗时（毫秒）

## 2. A-B 对比结果

| 实验组 | Recall@5 | MRR | Hit Rate@5 | 幻觉率 | 平均延迟(ms) | 样本 |
| --- | --- | --- | --- | --- | --- | --- |
| 仅向量检索 | 65.0% | 0.592 | 65.0% | 0.5% | 34.0 | 20 |
| 仅 BM25 | 70.0% | 0.562 | 70.0% | 0.5% | 0.0 | 20 |
| 混合检索(RRF) | 70.0% | 0.613 | 70.0% | 0.5% | 23.0 | 20 |
| 混合检索 + LLM重排 | 70.0% | 0.650 | 70.0% | 0.5% | 9477.0 | 20 |

## 3. 工具调用 A/B 对比（显式检索 vs Function Calling）

- 样本：15 条；成本为 `tiktoken` 近似计数，用于比较**相对**开销
- 轮次上限：`TOOL_CALLING_MAX_STEPS=3`

| 路径 | 成功率 | 平均延迟(ms) | P95 延迟(ms) | 平均 token | 合计 token | 平均轮次 | 平均工具调用 | 工具调用率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 显式检索 | 100.0% | 35422 | 50934 | 3286 | 49285 | 1.00 | 1.00 | 100.0% |
| Function Calling | 100.0% | 33043 | 48219 | 1995 | 29919 | 3.00 | 2.00 | 93.3% |

### 结论（全部来自上表实测）

- 延迟：显式检索 35422ms → 工具调用 33043ms（减少 2379ms）；P95 从 50934ms 变为 48219ms，长尾是工具调用路径最需要关注的成本。
- 成本：平均 token 从 3286 变为 1995（约 0.61x）；多出的部分主要来自「决策轮 + 工具结果回灌」，而非最终答案变长。
- 自主性：模型在 93.3% 的样本上主动调用了工具，平均 2.00 次/样本；显式路径固定 1 次。差距说明工具调用只在「模型判断确有必要」时才付出额外开销。
- 稳定性：成功率 显式 100.0% vs 工具调用 100.0%（失败样本 0 / 0 条）。
- 取舍建议：两条路径成本接近时，优先用工具调用以换取更强的自主性；若延迟敏感则保留显式检索。

## 4. 轨迹级评测（TaskSuccessRate / Failure Onset）

口径：成功由场景结构标记与工具事件判定（不额外调用模型，可复现）；
「依赖检索的场景」出现工具降级时即便文本工整也判为失败。

- **TaskSuccessRate**：100.0%（15/15）
- 工具失败率：20.0%（有工具降级的样本占比）
- 达到轮次上限的比例：6.7%
- 平均轮次：3.0
- **Failure Onset 均值**：2.0（越小说明越早出错，越该先修前面的环节）
- 对照组（显式检索，15 条）：工具失败率 20.0%，Failure Onset 均值 1.0

Failure Onset 直方图（第几步）：

- 第 2 步：3 条

最早失败示例：

- qa 第 2 步：knowledge_search 失败（timeout）
- qa 第 2 步：knowledge_search 失败（timeout）
- qa 第 2 步：knowledge_search 失败（timeout）

## 5. 结论与优化建议

- 单路检索中 **仅 BM25** 的 Recall@K 为 70.0%；混合检索（RRF）为 70.0%，相对最优单路 提升 0.0 个百分点。本次样本量下两者持平（差异不足 0.5 个百分点）。小知识库容易让 Recall 触顶、掩盖两路召回的差异；样本量或知识库变大后这项差异才有统计意义，届时再决定是否默认开启混合检索。
- 引入 LLM 重排后 MRR 从 0.613 变为 0.650（提升 0.037），但平均延迟从 23.0ms 增加到 9477.0ms。重排是典型的「用延迟换准确率」，可按业务对时延的容忍度开关。
- 延迟区间 0.0ms（仅 BM25）~ 9477.0ms（混合检索 + LLM重排）。线上可通过结果缓存 + 缩小 rerank_top_n 进一步压缩。
- 后续可继续做网格实验：chunk_size ∈ {300, 500, 800}、top_k ∈ {3, 5, 10}，用同一份评测集跑出最优组合（修改 `RetrievalConfig` 即可复现）。

## 6. 配置快照

```json
{
  "model": "deepseek-v4.1-flash",
  "embedding": "local/BAAI/bge-small-zh-v1.5",
  "chunk": {
    "size": 500,
    "overlap": 80
  },
  "experiments": [
    {
      "name": "仅向量检索",
      "config": {
        "use_bm25": false,
        "use_vector": true,
        "fusion": "vector_only",
        "top_k": 5,
        "reranker": "none",
        "rerank_top_n": 20,
        "chunk_size": 500,
        "chunk_overlap": 80
      }
    },
    {
      "name": "仅 BM25",
      "config": {
        "use_bm25": true,
        "use_vector": false,
        "fusion": "bm25_only",
        "top_k": 5,
        "reranker": "none",
        "rerank_top_n": 20,
        "chunk_size": 500,
        "chunk_overlap": 80
      }
    },
    {
      "name": "混合检索(RRF)",
      "config": {
        "use_bm25": true,
        "use_vector": true,
        "fusion": "rrf",
        "top_k": 5,
        "reranker": "none",
        "rerank_top_n": 20,
        "chunk_size": 500,
        "chunk_overlap": 80
      }
    },
    {
      "name": "混合检索 + LLM重排",
      "config": {
        "use_bm25": true,
        "use_vector": true,
        "fusion": "rrf",
        "top_k": 5,
        "reranker": "llm",
        "rerank_top_n": 20,
        "chunk_size": 500,
        "chunk_overlap": 80
      }
    }
  ],
  "tool_comparison": {
    "sample_count": 15,
    "explicit": {
      "name": "显式检索",
      "samples": 15,
      "failures": 0,
      "success_rate": 1.0,
      "avg_latency_ms": 35421.87,
      "p95_latency_ms": 50934.0,
      "avg_tokens": 3285.67,
      "total_tokens": 49285,
      "avg_rounds": 1.0,
      "avg_tool_calls": 1.0,
      "tool_call_rate": 1.0,
      "config": {
        "path": "explicit",
        "model": "deepseek-v4.1-flash",
        "retrieval": {}
      }
    },
    "tool_calling": {
      "name": "Function Calling",
      "samples": 15,
      "failures": 0,
      "success_rate": 1.0,
      "avg_latency_ms": 33043.33,
      "p95_latency_ms": 48219.0,
      "avg_tokens": 1994.6,
      "total_tokens": 29919,
      "avg_rounds": 3.0,
      "avg_tool_calls": 2.0,
      "tool_call_rate": 0.9333,
      "config": {
        "path": "tool_calling",
        "model": "deepseek-v4.1-flash",
        "max_steps": 3,
        "tools": [
          "knowledge_search"
        ]
      }
    }
  },
  "trajectory": {
    "samples": 15,
    "success": 15,
    "success_rate": 1.0,
    "failed_reasons": {},
    "tool_failure_rate": 0.2,
    "over_step_rate": 0.0667,
    "avg_rounds": 3.0,
    "failure_onset_hist": {
      "2": 3
    },
    "avg_failure_onset": 2.0,
    "onset_examples": [
      "qa 第 2 步：knowledge_search 失败（timeout）",
      "qa 第 2 步：knowledge_search 失败（timeout）",
      "qa 第 2 步：knowledge_search 失败（timeout）"
    ],
    "explicit_degradation": {
      "samples": 15,
      "tool_failure_rate": 0.2,
      "avg_failure_onset": 1.0,
      "failure_onset_hist": {
        "1": 3
      },
      "note": "对照路径按「检索是否降级」统计，不判定答案成功率"
    }
  }
}
```
