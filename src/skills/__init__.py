"""Skills 骨架：注册 / 版本 / 权限（详见 `registry` 模块 docstring）。

对外只暴露几个动作，避免调用方绕过注册表直接摸内部结构：
- `register_defaults()`：启动时注册项目自带场景（幂等）；
- `effective_permissions(mode)` / `has_permission(mode, perm)`：装配工具与校验能力；
- `snapshot()`：`/health` 展示当前已注册技能与授权情况。
"""

from .registry import (
    PERMISSION_LABELS,
    REGISTRY,
    SKILL_API_VERSION,
    Skill,
    SkillRegistry,
    effective_permissions,
    granted_permissions,
    has_permission,
    register_defaults,
    snapshot,
)

__all__ = [
    "PERMISSION_LABELS",
    "REGISTRY",
    "SKILL_API_VERSION",
    "Skill",
    "SkillRegistry",
    "effective_permissions",
    "granted_permissions",
    "has_permission",
    "register_defaults",
    "snapshot",
]
