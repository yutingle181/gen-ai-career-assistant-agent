"""LangGraph 节点与路由函数。

职责边界：
- 分类节点：调用 LLM 做一级/二级分类（结构化输出 + 关键词兜底）；
- 叶子节点：只负责「确定模式 + 给出首轮引导语」，真正的生成由 SessionManager 驱动，
  这样图既保留了可可视化的完整拓扑，又不会在 Web 环境里阻塞。
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from ..llm import llm_invoke
from ..logging_setup import get_logger, summarize
from ..models import CategoryResult, InterviewSubResult, LearningSubResult
from ..prompts.classify import (
    CATEGORIZE_PROMPT,
    FALLBACK_MESSAGE,
    INTERVIEW_SUB_PROMPT,
    LEARNING_SUB_PROMPT,
)
from ..rag.registry import get_registry
from ..state import (
    CAT_FALLBACK,
    CAT_INTERVIEW,
    CAT_JOB_SEARCH,
    CAT_KNOWLEDGE,
    CAT_LEARNING,
    CAT_RESUME,
    MODE_FALLBACK,
    MODE_INTERVIEW_QUESTIONS,
    MODE_JOB_SEARCH,
    MODE_KNOWLEDGE,
    MODE_LABELS,
    MODE_MOCK_INTERVIEW,
    MODE_QA,
    MODE_RESUME,
    MODE_TUTORIAL,
    State,
)
from ..telemetry import span

logger = get_logger(__name__)


# ---------------------------------------------------------------- 分类工具
def _structured(prompt: str, query: str, schema):
    """优先结构化输出，失败时回退为纯文本解析。"""
    with span("graph.classify", {"graph.query_chars": len(query)}) as sp:
        try:
            from ..llm import get_chat_model

            model = get_chat_model(temperature=0).with_structured_output(schema)
            result = model.invoke([HumanMessage(content=prompt.format(query=query))])
            if result is not None:
                sp.set_attribute("graph.structured", True)
                return result
        except Exception as exc:  # noqa: BLE001
            sp.set_attribute("graph.structured", False)
            logger.debug("结构化分类失败，回退文本解析：%s", exc)

        try:
            raw = llm_invoke([HumanMessage(content=prompt.format(query=query))], temperature=0)
        except Exception as exc:  # noqa: BLE001
            sp.set_attribute("error", True)
            sp.set_attribute("error.type", type(exc).__name__)
            logger.warning("分类调用失败：%s", exc)
            return None
        sp.set_attribute("graph.structured", False)
        return _parse_text(raw, schema)


def _parse_text(raw: str, schema):
    """从文本中抽取第一个合法枚举值。"""
    text = (raw or "").strip().lower()
    fields = list(schema.model_fields)
    if not fields:
        return None
    target = fields[0]
    allowed = schema.model_fields[target].annotation
    values = []
    for v in getattr(allowed, "__args__", ()):  # Literal[...]
        values.append(str(v).lower())
    for v in values:
        if re.search(rf"\b{re.escape(v)}\b", text):
            return schema(**{target: v})
    return None


# ---------------------------------------------------------------- 一级分类
def categorize(state: State) -> dict:
    """一级分类：学习 / 简历 / 面试 / 求职 / 知识库。"""
    query = state.get("query", "")
    logger.info("一级分类 | %s", summarize(query))

    result = _structured(CATEGORIZE_PROMPT, query, CategoryResult)
    category = result.category if isinstance(result, CategoryResult) else None
    reason = result.reason if isinstance(result, CategoryResult) else ""

    if category is None:
        category, reason = _heuristic_category(query)
    elif category == CAT_KNOWLEDGE and not _has_knowledge_signal(query):
        # 判为知识库问答，却没有任何指向已上传资料的措辞：
        # 此时进知识库节点只会得到「尚未创建知识库」，不如退回启发式给通用回答。
        logger.info("知识库分类缺少资料指向，退回启发式 | %s", summarize(query))
        category, reason = _heuristic_category(query)

    logger.info("一级分类结果 | %s", category)
    return {"category": category}


_KNOWLEDGE_SIGNALS = ("知识库", "上传", "文档", "资料", "pdf", "文件", "笔记", "我上传")


def _has_knowledge_signal(query: str) -> bool:
    """输入里是否出现指向「已上传资料」的措辞。"""
    q = (query or "").lower()
    return any(k in q for k in _KNOWLEDGE_SIGNALS)


def _heuristic_category(query: str) -> tuple[str, str]:
    """关键词兜底：LLM 不可用或输出非法时保证图永远有出口。"""
    q = (query or "").lower()
    rules = [
        (CAT_KNOWLEDGE, ("知识库", "上传的文档", "资料里", "根据文档", "我上传", "简历里")),
        (CAT_JOB_SEARCH, ("招聘", "职位", "岗位", "job", "找工作", "求职", "在招", "招人")),
        (CAT_RESUME, ("简历", "resume", "cv")),
        (CAT_INTERVIEW, ("面试", "interview", "面经")),
        (CAT_LEARNING, ("教程", "tutorial", "学习", "怎么学", "博客", "langchain", "langgraph", "rag")),
    ]
    for cat, keywords in rules:
        if any(k in q for k in keywords):
            return cat, "关键词兜底命中"
    return CAT_LEARNING, "默认归类为学习问答"


# ---------------------------------------------------------------- 二级分类
def handle_learning_resource(state: State) -> dict:
    """学习类二级分类：教程 or 问答。"""
    result = _structured(LEARNING_SUB_PROMPT, state.get("query", ""), LearningSubResult)
    if isinstance(result, LearningSubResult):
        return {"category": result.sub}
    # 离线/LLM 不可用兜底：用关键词判断二级分类
    q = (state.get("query", "") or "").lower()
    if any(k in q for k in ("教程", "tutorial", "写一份", "生成", "博客", "blog", "上手", "入门")):
        return {"category": "tutorial"}
    return {"category": "question"}


def handle_interview_preparation(state: State) -> dict:
    """面试类二级分类：模拟面试 or 真题。

    意图明确时先用确定性规则判定，不依赖模型：
    不同模型（尤其不同快照版本）对「整理 20 道面试题」这类说法的判断并不一致，
    而这类说法在界面示例与真实输入里都很常见，交给模型容易误判为模拟面试。
    """
    q = (state.get("query", "") or "").lower()

    # 明确要「被提问 / 练习」→ 模拟面试
    if any(k in q for k in ("模拟面试", "mock", "面试练习", "面我", "来面", "考我", "练一练", "陪我练")):
        return {"category": "mock"}

    # 明确要「整理 / 列出」一批题目 → 真题清单
    list_intent = ("整理", "列出", "汇总", "真题", "题库", "清单", "套题")
    if any(k in q for k in list_intent) or re.search(r"\d+\s*道", q):
        return {"category": "question"}

    result = _structured(INTERVIEW_SUB_PROMPT, state.get("query", ""), InterviewSubResult)
    if isinstance(result, InterviewSubResult):
        return {"category": result.sub}
    if any(k in q for k in ("模拟面试", "mock", "面试练习", "面我", "来面", "考我", "练一练")):
        return {"category": "mock"}
    return {"category": "question"}


# ---------------------------------------------------------------- 叶子节点
def _leaf(mode: str, extra: str = "") -> dict:
    label = MODE_LABELS.get(mode, mode)
    message = f"已识别为【{label}】，正在为你处理…"
    if extra:
        message += f"\n\n{extra}"
    return {"mode": mode, "message": message}


def tutorial_agent(state: State) -> dict:
    return _leaf(MODE_TUTORIAL)


def ask_query_bot(state: State) -> dict:
    return _leaf(MODE_QA)


def handle_resume_making(state: State) -> dict:
    return _leaf(MODE_RESUME, "我会分 4-5 步询问你的技能、经历与项目，最后生成 Markdown 简历。")


def interview_topics_questions(state: State) -> dict:
    return _leaf(MODE_INTERVIEW_QUESTIONS)


def mock_interview(state: State) -> dict:
    return _leaf(MODE_MOCK_INTERVIEW, "我将扮演面试官逐题提问，你随时可以输入「结束」获取面评。")


def job_search(state: State) -> dict:
    return _leaf(MODE_JOB_SEARCH, "提示：在问题里写清**城市**和**岗位**，检索结果会更准。")


def knowledge_qa(state: State) -> dict:
    registry = get_registry()
    names = registry.names()
    if not names:
        return _leaf(MODE_KNOWLEDGE, "尚未创建知识库，请先在左侧上传文档并完成建库。")
    return _leaf(MODE_KNOWLEDGE, f"当前可选知识库：{', '.join(names)}")


def fallback(state: State) -> dict:
    """兜底节点：保证图永远有出口。"""
    return {"mode": MODE_FALLBACK, "message": FALLBACK_MESSAGE, "category": CAT_FALLBACK}


# ---------------------------------------------------------------- 路由
def route_query(state: State) -> str:
    """一级路由。"""
    category = (state.get("category") or "").lower()
    mapping = {
        CAT_LEARNING: "handle_learning_resource",
        CAT_RESUME: "handle_resume_making",
        CAT_INTERVIEW: "handle_interview_preparation",
        CAT_JOB_SEARCH: "job_search",
        CAT_KNOWLEDGE: "knowledge_qa",
    }
    if category in mapping:
        logger.info("路由 | %s -> %s", category, mapping[category])
        return mapping[category]
    logger.warning("未知一级分类 %s，走兜底节点", category)
    return "fallback"


def route_learning(state: State) -> str:
    """学习类二级路由。"""
    sub = (state.get("category") or "").lower()
    if "tutorial" in sub:
        return "tutorial_agent"
    if "question" in sub:
        return "ask_query_bot"
    logger.warning("未知学习子类 %s，默认走问答", sub)
    return "ask_query_bot"


def route_interview(state: State) -> str:
    """面试类二级路由。"""
    sub = (state.get("category") or "").lower()
    if "mock" in sub:
        return "mock_interview"
    if "question" in sub:
        return "interview_topics_questions"
    logger.warning("未知面试子类 %s，默认走真题", sub)
    return "interview_topics_questions"
