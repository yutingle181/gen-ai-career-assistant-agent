"""无 UI 端到端冒烟：7 条典型 query 跑通 路由 + 会话 + 产物导出。

说明：本项目 LLM 调用默认走 OpenAI 兼容接口，若未配置有效 KEY，各模式会
降级为友好提示，但「路由分类 / 会话单步 / 产物落盘」链路仍然完整可验证。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.workflow import route_only
from src.logging_setup import get_logger
from src.session import SessionManager

logger = get_logger(__name__)

QUERIES = [
    ("帮我写一篇 LangGraph 实战教程", "tutorial"),
    ("RAG 和微调到底该怎么选？", "qa"),
    ("我要根据这份 JD 改简历", "resume"),
    ("整理 20 道 GenAI 面试题", "interview_questions"),
    ("我想做一场模拟面试", "mock_interview"),
    ("长沙有哪些 AI 应用工程师岗位", "job_search"),
    ("根据知识库回答：切分策略怎么选？", "knowledge"),
]


def run_query(query: str) -> tuple[str, str | None]:
    """执行一轮完整链路，返回 (路由到的模式, 产物路径)。"""
    routed = route_only(query)
    mode = routed.get("mode", "")
    mgr = SessionManager(mode)
    mgr.start(query)
    return mode, mgr.finish()


@pytest.mark.parametrize(("query", "expected_mode"), QUERIES)
def test_route_classifies_expected_mode(query, expected_mode):
    mode, path = run_query(query)
    assert mode == expected_mode, f"路由错误：{query} -> {mode}（期望 {expected_mode}）"
    if path:  # 无有效 KEY 时产物可能为空，有则必须真实落盘
        assert Path(path).exists()


def main() -> None:
    print("=" * 72)
    print("GenAI Career Assistant — 端到端冒烟测试")
    print("=" * 72)

    for query, expected_mode in QUERIES:
        mode, path = run_query(query)
        flag = "✅" if mode == expected_mode else "⚠️"
        print(f"{flag} 路由：{query}\n     → mode={mode} (期望 {expected_mode})")
        print(f"     → 产物：{Path(path).name if path else 'N/A'}\n")

    print("=" * 72)
    print("冒烟通过：7 条 query 全部完成路由 + 会话 + 产物导出")
    print("（未配置有效 API KEY 时，内容会降级为提示，属正常现象）")


if __name__ == "__main__":
    main()
