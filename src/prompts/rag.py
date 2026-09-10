"""RAG 相关 Prompt：查询改写、评测集生成、幻觉判定。"""

from __future__ import annotations

QUERY_REWRITE_PROMPT = """你是一个检索查询改写助手。把用户的提问改写为更适合关键词与语义检索的检索式。
要求：
1. 输出 1 条改写后的查询，不要解释；
2. 保留专有名词原样（如 LangGraph、bge-m3、RAG）；
3. 如果是多轮对话中的追问，结合上下文补全指代。

对话历史：
{history}

用户最新提问：{query}

改写后的检索式："""

EVAL_GEN_PROMPT = """你是一个评测集生成助手。下面是一段知识库文档片段，请基于它生成 {n} 个问答对。
要求：
1. 问题必须能由该片段直接回答，不要超出片段内容；
2. 答案简洁（1-3 句），必须是片段中事实的复述；
3. 只输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出其他内容。

文档片段：
{chunk}

JSON："""

HALLUCINATION_JUDGE_PROMPT = """你是一个事实一致性评判员。给定【召回资料】与【模型回答】，
请把回答拆成若干条可验证的断言，逐条判断该断言是否能由资料支撑。

输出 JSON：{{"claims": [{{"text": "断言内容", "supported": true}}]}}
只输出 JSON，不要解释。

【召回资料】
{context}

【模型回答】
{answer}

JSON："""

ANSWER_WITH_CITATION_HINT = "（回答中请用 [1]、[2] 标注引用来源编号）"
