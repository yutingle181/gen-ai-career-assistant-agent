"""Skills 骨架：注册 / 版本 / 权限。

背景：此前「按 mode 加载人设与工具」靠的是一个裸字典 `AGENT_CLASSES` 加上各 Agent 类里
硬编码的 `needs_*` 布尔量，于是三件事都缺：

1. **注册**：有哪些场景、各自什么版本、谁能改，只能靠翻代码；
2. **版本**：替换某场景实现时，没有任何兼容性契约可校验（换了个签名不匹配的类也照跑）；
3. **权限**：任何场景都可能挂到全部工具（联网 / 知识库 / MCP），"最小权限"无从表达。

这里补的是**骨架**，不是插件市场、也不做热加载：

- 每个场景声明为一条 `Skill`（名字 / 版本 / 接口版本 / 模式 / Agent 类 / **需要的权限**）；
- 全局只授予 `config.SKILL_PERMISSIONS` 里列出的权限，**生效权限 = 声明 ∩ 授予**；
- 装配工具时按生效权限过滤（没授权就不挂载），显式检索路径同样校验；
- 接口主版本不兼容时默认**拒绝注册**并回退默认场景 —— 宁可少一个场景，
  也不要带着一份不确定的契约跑生产。

能力边界（必须如实说）：权限是**装配期授权**（不挂载 / 不执行），**不是沙箱**；
工具一旦挂上，运行期能做什么取决于工具实现与操作系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import config, metrics
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: 技能接口版本：Agent 类必须满足的契约（`mode` / `respond` / `respond_stream` /
#: `build_tools` / `prepare`）。接口做不兼容改动时递增**主版本**。
SKILL_API_VERSION = "1.0"

#: 权限字典：能力 → 人类可读说明（拒绝挂载时写进日志，便于排查"为什么工具没了"）
PERMISSION_LABELS: dict[str, str] = {
    "net": "联网检索",
    "kb": "内部知识库检索",
    "mcp": "外部 MCP 工具",
}


@dataclass(frozen=True)
class Skill:
    """一个场景能力（skill）的声明。"""

    name: str
    mode: str
    agent_cls: type
    version: str = "1.0.0"
    api_version: str = SKILL_API_VERSION
    title: str = ""
    description: str = ""
    #: 该场景**需要**的权限；实际生效 = 声明 ∩ config.SKILL_PERMISSIONS
    permissions: frozenset[str] = field(default_factory=frozenset)
    enabled: bool = True

    @property
    def major(self) -> int:
        """接口主版本（用于兼容性判断）。解析不出来时按 0 处理（会被判为不兼容）。"""
        head = str(self.api_version).split(".", 1)[0]
        return int(head) if head.isdigit() else 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "version": self.version,
            "api_version": self.api_version,
            "title": self.title,
            "permissions": sorted(self.permissions),
            "enabled": self.enabled,
            "agent": getattr(self.agent_cls, "__name__", str(self.agent_cls)),
        }


@dataclass
class RegistrationResult:
    """注册结果（拒绝的原因要能说清楚，否则排查只能靠猜）。"""

    ok: bool
    reason: str = ""


class SkillRegistry:
    """技能注册表：注册 / 查询 / 权限计算（进程内，启动时一次性填充）。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._by_mode: dict[str, str] = {}
        self._rejected: dict[str, str] = {}

    # ------------------------------------------------------------ 注册
    def register(self, skill: Skill) -> RegistrationResult:
        """注册一个 skill；同模式重复注册、接口不兼容、缺契约一律拒绝。"""
        if not skill.name or "." not in skill.version:
            return self._reject(skill.name or "?", "name 为空或 version 不是 x.y.z 形式")
        if skill.major != _interface_major() and not config.SKILL_ALLOW_API_MISMATCH:
            return self._reject(
                skill.name,
                f"接口版本不兼容：声明 {skill.api_version}，当前接口 {SKILL_API_VERSION}"
                "（如确需宽容处理，设置 SKILL_ALLOW_API_MISMATCH=true）",
            )
        missing = [attr for attr in ("mode", "respond", "respond_stream") if not hasattr(skill.agent_cls, attr)]
        if missing:
            return self._reject(skill.name, f"Agent 类缺少契约成员：{', '.join(missing)}")
        if skill.name in self._skills:
            return self._reject(skill.name, "同名 skill 已注册")

        previous = self._by_mode.get(skill.mode)
        self._skills[skill.name] = skill
        self._by_mode[skill.mode] = skill.name
        if previous:
            logger.warning("模式 %s 的 skill 被替换：%s → %s", skill.mode, previous, skill.name)
        metrics.incr("skill.registered")
        return RegistrationResult(True)

    def _reject(self, name: str, reason: str) -> RegistrationResult:
        self._rejected[name] = reason
        metrics.incr("skill.register_rejected")
        logger.warning("skill 注册被拒 | %s | %s", name, reason)
        return RegistrationResult(False, reason)

    # ------------------------------------------------------------ 查询
    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def for_mode(self, mode: str) -> Skill | None:
        """按模式取 skill；被禁用的视为不存在（调用方负责回退）。"""
        name = self._by_mode.get(mode)
        skill = self._skills.get(name) if name else None
        if skill is not None and not skill.enabled:
            logger.info("skill 已禁用，按未注册处理 | %s", skill.name)
            return None
        return skill

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def modes(self) -> list[str]:
        return sorted(self._by_mode)

    def snapshot(self) -> dict[str, Any]:
        granted = granted_permissions()
        return {
            "api_version": SKILL_API_VERSION,
            "granted_permissions": sorted(granted),
            "skills": [s.snapshot() for s in self.all()],
            "rejected": dict(self._rejected),
        }

    def reset(self) -> None:
        """清空注册表（测试用）。"""
        self._skills.clear()
        self._by_mode.clear()
        self._rejected.clear()

    # ------------------------------------------------------------ 权限
    def effective_permissions(self, mode: str) -> frozenset[str]:
        """生效权限 = skill 声明 ∩ 全局授予。未注册的模式不给任何权限。"""
        skill = self.for_mode(mode)
        if skill is None:
            return frozenset()
        return frozenset(skill.permissions & granted_permissions())


def _interface_major() -> int:
    head = str(SKILL_API_VERSION).split(".", 1)[0]
    return int(head) if head.isdigit() else 0


REGISTRY = SkillRegistry()


def register_defaults() -> None:
    """把项目自带的 9 个场景注册进来（幂等）。

    放在这里而不是散落各处：**这张表就是「项目有哪些能力」的唯一事实来源**，
    与 `agents.AGENT_CLASSES`（由本表派生）保持同源，避免两处各写一份。
    权限按各场景的**真实需要**声明，而不是"能给的都给"：
    知识库问答只给 `kb`（联网结果会污染引用来源），职位检索才同时要 `net` + `kb`。
    """
    from ..agents import (
        InterviewQuestionsAgent,
        InterviewReviewAgent,
        JDMatchAgent,
        JobSearchAgent,
        KnowledgeAgent,
        MockInterviewAgent,
        QAAgent,
        ResumeAgent,
        TutorialAgent,
    )
    from ..state import (
        MODE_INTERVIEW_QUESTIONS,
        MODE_INTERVIEW_REVIEW,
        MODE_JD_MATCH,
        MODE_JOB_SEARCH,
        MODE_KNOWLEDGE,
        MODE_MOCK_INTERVIEW,
        MODE_QA,
        MODE_RESUME,
        MODE_TUTORIAL,
    )

    defaults = [
        Skill(
            name="learning.tutorial",
            mode=MODE_TUTORIAL,
            agent_cls=TutorialAgent,
            version="1.0.0",
            title="教程生成",
            description="联网检索后一次性产出 Markdown 教程（一次性场景）。",
            permissions=frozenset({"net"}),
        ),
        Skill(
            name="qa.general",
            mode=MODE_QA,
            agent_cls=QAAgent,
            version="1.0.0",
            title="答疑问答",
            description="多轮追问的通用答疑；不依赖外部资料。",
        ),
        Skill(
            name="resume.builder",
            mode=MODE_RESUME,
            agent_cls=ResumeAgent,
            version="1.0.0",
            title="简历制作",
            description="4-5 步收集信息生成简历；产物外发前需人工确认。",
        ),
        Skill(
            name="interview.questions",
            mode=MODE_INTERVIEW_QUESTIONS,
            agent_cls=InterviewQuestionsAgent,
            version="1.0.0",
            title="面试真题",
            description="检索后一次性生成题库。",
            permissions=frozenset({"net"}),
        ),
        Skill(
            name="interview.mock",
            mode=MODE_MOCK_INTERVIEW,
            agent_cls=MockInterviewAgent,
            version="1.0.0",
            title="模拟面试",
            description="一问一答，结束时结构化输出面评。",
        ),
        Skill(
            name="job.search",
            mode=MODE_JOB_SEARCH,
            agent_cls=JobSearchAgent,
            version="1.0.0",
            title="职位检索",
            description="联网检索职位并整理为结构化清单；含外链，需人工确认。",
            permissions=frozenset({"net", "kb"}),
        ),
        Skill(
            name="knowledge.qa",
            mode=MODE_KNOWLEDGE,
            agent_cls=KnowledgeAgent,
            version="1.0.0",
            title="知识库问答",
            description="带引用的内部资料问答；只允许查内部资料。",
            permissions=frozenset({"kb"}),
        ),
        Skill(
            name="jd.match",
            mode=MODE_JD_MATCH,
            agent_cls=JDMatchAgent,
            version="1.0.0",
            title="JD 匹配诊断",
            description="结合岗位 JD 与简历输出评分卡；需人工确认后定稿。",
        ),
        Skill(
            name="interview.review",
            mode=MODE_INTERVIEW_REVIEW,
            agent_cls=InterviewReviewAgent,
            version="1.0.0",
            title="面试复盘",
            description="结构化复盘：评分 / 追问链 / 薄弱点 / 改进动作。",
        ),
    ]

    disabled = {
        item.strip() for item in str(getattr(config, "SKILLS_DISABLED", "") or "").split(",") if item.strip()
    }
    registered = 0
    for skill in defaults:
        if skill.name in disabled:
            metrics.incr("skill.disabled")
            logger.warning("skill 被配置禁用，该模式将回退默认场景 | %s", skill.name)
            continue
        result = REGISTRY.register(skill)
        if result.ok:
            registered += 1
    logger.info(
        "默认 skill 注册完成 | 成功 %d / 共 %d | 权限授予 %s",
        registered,
        len(defaults),
        sorted(granted_permissions()),
    )


def granted_permissions() -> frozenset[str]:
    """全局授予的权限（配置驱动；默认与改动前的能力面一致，保证零回归）。"""
    raw = str(getattr(config, "SKILL_PERMISSIONS", "") or "")
    items = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return frozenset(items)


def effective_permissions(mode: str) -> frozenset[str]:
    """按模式取生效权限（供 Agent 装配工具、校验显式检索用）。"""
    return REGISTRY.effective_permissions(mode)


def has_permission(mode: str, permission: str) -> bool:
    return permission in effective_permissions(mode)


def snapshot() -> dict[str, Any]:
    """注册表快照（/health 用）。"""
    return REGISTRY.snapshot()
