"""上云前置：探活端点、安全基线（CORS / 鉴权提示）。

这三件事在本地开发时都看不出问题，只有"要挂到公网"时才会暴露，
所以用测试把语义钉住，避免以后改配置时无声退化。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import config
from src.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


# ----------------------------------------------------------------- 探活端点
def test_healthz_is_zero_dependency():
    """`/healthz` 必须零外部依赖：它要能在没 Key、没模型、没网络时也返回 200。

    为什么不能拿 `/health` 做探活（docker-compose / nginx / k8s 都会用）：
    `/health` 会真实调一次 LLM + 一次 Embedding，于是
    ①每 30 秒烧一次 token；②LLM 超时直接让探活失败 → 容器被反复重启，越探测越坏；
    ③没配 Key 时它仍返回 200 + degraded，探活区分不出"活着"和"不可用"。
    """
    with _client() as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"], "应带上版本号，便于确认线上跑的是哪次构建"


def test_healthz_needs_no_api_key(monkeypatch):
    """探活**不能**受 API_KEY 影响：否则配上鉴权后健康检查直接 401，容器永远起不来。"""
    monkeypatch.setattr(config, "API_KEY", "secret-key-for-test")
    with _client() as c:
        r = c.get("/healthz")
    assert r.status_code == 200, "探活端点必须豁免鉴权"


def test_root_advertises_healthz():
    """`/` 里要能看到 /healthz 的存在（否则没人知道有这个端点）。"""
    with _client() as c:
        body = c.get("/").json()
    assert body["healthz"] == "/healthz"


# ----------------------------------------------------------------- 安全基线
def test_cors_origins_are_explicit_not_wildcard(monkeypatch):
    """CORS 必须是显式白名单。

    `allow_origins=["*"]` 与 `allow_credentials=True` 是**非法组合**，浏览器会直接拒掉
    这类响应 —— 也就是原来的配置在跨域场景下根本不生效，属于"配了但没用"的假配置。
    """
    from src.api.main import CORSMiddleware  # noqa: F401 - 确认依赖已装载

    with _client() as c:
        r = c.options(
            "/chat",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    # 恶意来源不应拿到 allow-origin 头
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_cors_allows_configured_origin(monkeypatch):
    """白名单里的来源要能正常拿到 CORS 头（否则前端会跨域失败）。"""
    monkeypatch.setattr(config, "CORS_ORIGINS", ["https://my-frontend.example"])
    with _client() as c:
        r = c.options(
            "/chat",
            headers={
                "Origin": "https://my-frontend.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert r.headers.get("access-control-allow-origin") == "https://my-frontend.example"


def test_public_exposure_risks_flags_missing_api_key(monkeypatch):
    """没设 API_KEY 必须被点名为风险项 —— 这是最容易被忽略、代价最大的一条。"""
    monkeypatch.setattr(config, "API_KEY", "")
    risks = config.public_exposure_risks()
    assert any("API_KEY" in r for r in risks), risks


def test_public_exposure_risks_clean_when_configured(monkeypatch):
    """配置到位时不应误报（有告警等于没告警，久而久之所有人都会忽略它）。"""
    monkeypatch.setattr(config, "API_KEY", "k")
    monkeypatch.setattr(config, "API_RATE_LIMIT", 60)
    monkeypatch.setattr(config, "CORS_ORIGINS", ["https://my-frontend.example"])
    assert config.public_exposure_risks() == []


def test_health_exposes_security_block(monkeypatch):
    """`/health` 里要能看到鉴权状态与风险项（部署后点一次就知道该改什么）。"""
    with _client() as c:
        runtime = c.get("/health").json()["runtime"]
    assert "auth_enabled" in runtime
    assert isinstance(runtime["public_exposure_risks"], list)
