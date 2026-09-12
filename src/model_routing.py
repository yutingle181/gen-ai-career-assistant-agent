"""模型路由：按请求复杂度选择强 / 快模型（9.2.A）。

仅当强模型为 qwen 家族（生产 qwen-plus）且 ENABLE_MODEL_ROUTING 开启时生效；
保守默认走强模型（MODEL_STRONG），仅对明确简单的请求降级到 MODEL_FAST(qwen-turbo)。
路由判定纯本地、零网络、零额外延迟，与网关 ChatResponseCache / 召回 ResultCache /
Prompt 缓存三层互补、互不影响。
"""

from __future__ import annotations

import re

from . import config

# 复杂意图关键词：命中任一即视为「复杂任务」，走强模型
_COMPLEX_KEYWORDS = (
    "写", "生成", "创作", "起草", "拟", "润色", "优化", "改进", "修改", "改写",
    "评估", "评测", "打分", "分析", "方案", "对比", "比较", "总结", "概括",
    "翻译", "代码", "编程", "脚本", "表格", "列表", "计划", "规划", "建议",
    "梳理", "归纳", "复盘", "拆解", "解释", "为什么", "怎么", "如何",
)

# 简单意图模式：问候、致谢、确认、闲聊等
_SIMPLE_PATTERNS = (
    r"^\s*$",                                          # 空输入
    r"^(你好|您好|hi|hello|在吗|在么|哈喽|嗨|嘿)",
    r"(谢谢|感谢|多谢|辛苦了|好的|ok|okay|收到|明白了|懂了|了解|清楚|晓得了)",
    r"(再见|拜拜|下次|晚安)",
)


def _is_simple_query(query: str) -> bool:
    """依据长度 + 关键词 + 简单模式判定是否为「简单」请求。"""
    q = (query or "").strip()
    if not q:
        return True
    if len(q) > config.ROUTING_SIMPLE_MAX_CHARS:
        return False
    # 命中复杂关键词 -> 非简单
    for kw in _COMPLEX_KEYWORDS:
        if kw in q:
            return False
    # 命中简单模式 -> 简单
    for pat in _SIMPLE_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return True
    # 短且无明显复杂意图：视为简单，以体现路由收益
    return True


def select_model(query: str, mode: str | None = None,
                 user_context: str | None = None, history=None) -> str:
    """返回本轮应使用的模型名（MODEL_FAST 或 MODEL_STRONG）。

    设计原则：保守默认走强模型，路由只做「降压」不做「冒险」。
    任何异常都回落到 MODEL_STRONG，绝不中断请求。
    """
    try:
        if not config.ENABLE_MODEL_ROUTING:
            return config.MODEL_STRONG
        # 仅 qwen 家族支持双模型路由（生产 qwen-plus / qwen-turbo）
        if not config.MODEL_STRONG.lower().startswith("qwen"):
            return config.MODEL_STRONG
        # 带 user_context（JD+简历重任务）：走强模型，保证质量
        if user_context:
            return config.MODEL_STRONG
        # 仅对明确的「简单对话答疑」模式做降级
        if mode not in config.ROUTING_SIMPLE_MODES:
            return config.MODEL_STRONG
        if _is_simple_query(query):
            return config.MODEL_FAST
        return config.MODEL_STRONG
    except Exception:  # noqa: BLE001
        return config.MODEL_STRONG
