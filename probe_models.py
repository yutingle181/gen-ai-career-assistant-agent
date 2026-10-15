"""模型 / 向量额度探测：一次性脚本，回答「现在到底还有哪个模型能用」。

背景（2026-10-15 实测）：`.env` 里钉的 `qwen-plus-latest` 与 `text-embedding-v3`
免费额度**同时耗尽**，`/health` 从 ok 退成 `degraded`（`llm=false`、`embedding=false`），
表现是「服务起来了但回答走降级」，而且**不报错、只在 /health 里体现**。
控制台里其实还有多个模型各剩百万级 tokens —— 所以需要「先探测、再切换」，
而不是凭印象改配置（改错模型名会让 /health 继续 degraded，还多烧一次排查）。

用法：
    .venv-ci\\Scripts\\python.exe probe_models.py
    .venv-ci\\Scripts\\python.exe probe_models.py --timeout 15 --json data/eval/probe.json

设计约束：
- **只用标准库**：即使虚拟环境本身有问题（依赖装不上、密钥失效）也能跑；
- **绝不打印密钥**：只报「是否配置 + 长度」；报错信息也不回显请求头；
- **最小请求**：会话用 ``max_tokens=1``、向量用 4 个字，尽量不烧额度；
- **逐行 flush**：本机长时间无输出的命令会被外部掐断，实时输出便于观察。

退出码：0 = 会话与向量都可用；10 = 会话可用但无可用向量；20 = 无可用会话模型。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 控制台额度页显示「各剩 1,000,000 / 共 1,000,000 tokens」的候选（2026-10-15 核对，未过期）
CHAT_CANDIDATES: list[tuple[str, str]] = [
    ("deepseek-v4.1-flash", "额度到 2026/12/13"),
    ("qwen3.8-flash", "额度到 2026/11/25"),
    ("qwen3.8-27b", "额度到 2026/11/18"),
    ("qwen3.8-max-0902", "额度到 2026/12/01"),
    ("glm-5.3", "额度到 2026/11/23"),
    ("kimi-k3", "额度到 2026/11/18"),
    ("deepseek-v4-flash-0731", "额度到 2026/10/31"),
    ("qwen-plus-latest", "当前 .env 取值（额度已耗尽，作为对照基线）"),
]
EMBEDDING_CANDIDATES: list[str] = [
    "text-embedding-v3",  # 当前 .env 取值（额度已耗尽，作为对照基线）
    "text-embedding-v4",
    "text-embedding-v2",
]


@dataclass
class ProbeResult:
    """单个候选模型的探测结论。"""

    kind: str  # chat | embedding
    model: str
    ok: bool
    status: str
    code: str = ""
    detail: str = ""


@dataclass
class ProbeReport:
    """一次探测的完整结论（含配置来源，便于事后复现）。"""

    chat_base: str = ""
    embedding_base: str = ""
    key_present: bool = False
    key_length: int = 0
    embedding_key_present: bool = False
    results: list[ProbeResult] = field(default_factory=list)

    def usable(self, kind: str) -> list[str]:
        return [r.model for r in self.results if r.kind == kind and r.ok]


def load_env(path: Path) -> dict[str, str]:
    """极简 .env 解析（不依赖 dotenv，保证裸环境可跑）。

    与 `src/config.py` 的差异：不处理多行值与 `export` 前缀 —— 本脚本只需要
    KEY / BASE_URL / MODEL 这类单行键，解析失败就当作「未配置」。
    """
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _post(url: str, api_key: str, payload: dict, timeout: int) -> tuple[str, str, str]:
    """POST 一个最小请求，返回 (status, code, detail)；成功时 status 为 `HTTP 200`。"""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.loads(resp.read().decode("utf-8", "replace"))  # 确认是合法 JSON 再算成功
            return f"HTTP {resp.status}", "", ""
    except urllib.error.HTTPError as exc:
        code, detail = "", ""
        try:
            payload_json = json.loads(exc.read().decode("utf-8", "replace"))
            err = payload_json.get("error", payload_json)
            code = str(err.get("code") or err.get("type") or "")
            detail = str(err.get("message") or payload_json.get("message") or "")
        except Exception:  # noqa: BLE001 - 供应商可能返回非 JSON 错误体
            detail = "non-json error body"
        return f"HTTP {exc.code}", code, detail[:160]
    except Exception as exc:  # noqa: BLE001 - 网络/超时/DNS 等一律归到这里
        return type(exc).__name__, "", str(exc)[:160]


def probe_chat(base: str, key: str, model: str, timeout: int) -> ProbeResult:
    status, code, detail = _post(
        base + "/chat/completions",
        key,
        {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        timeout,
    )
    return ProbeResult("chat", model, status == "HTTP 200", status, code, detail)


def probe_embedding(base: str, key: str, model: str, timeout: int) -> ProbeResult:
    status, code, detail = _post(base + "/embeddings", key, {"model": model, "input": "测试"}, timeout)
    return ProbeResult("embedding", model, status == "HTTP 200", status, code, detail)


def _print_table(kind: str, results: list[ProbeResult]) -> None:
    title = "会话模型" if kind == "chat" else "向量模型"
    print(f"=== {title} ===", flush=True)
    for r in results:
        mark = "OK  " if r.ok else "FAIL"
        note = r.detail if not r.ok and r.detail else (r.code if r.ok else "")
        print(f"  [{mark}] {r.model:<24} {r.status:<12} {note}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="探测 .env 配置的模型/向量端点当前是否可用（不打印密钥）")
    parser.add_argument("--timeout", type=int, default=12, help="单次请求超时秒数（默认 12）")
    parser.add_argument("--json", dest="json_out", default="", help="把结论写入该 JSON 文件（默认不写）")
    parser.add_argument("--chat", dest="chat_only", nargs="*", help="只探测指定会话模型")
    parser.add_argument("--embedding", dest="emb_only", nargs="*", help="只探测指定向量模型")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - 老解释器没有 reconfigure
        pass

    env = load_env(ROOT / ".env")
    if not env:
        print(f"[XX] 没找到 {ROOT / '.env'}，无法探测", file=sys.stderr)
        return 20
    key = env.get("OPENAI_API_KEY", "")
    base = env.get("OPENAI_BASE_URL", "").rstrip("/")
    emb_key = env.get("EMBEDDING_API_KEY", "") or key
    emb_base = (env.get("EMBEDDING_BASE_URL", "") or base).rstrip("/")
    if not key or not base:
        print("[XX] .env 缺少 OPENAI_API_KEY / OPENAI_BASE_URL", file=sys.stderr)
        return 20

    print("当前配置（只显示键名与长度，不显示值）：", flush=True)
    print(
        f"  chat : base={base} key={f'已设置({len(key)})' if key else '缺失'} MODEL_NAME={env.get('MODEL_NAME', '-')}",
        flush=True,
    )
    print(
        f"  embed: base={emb_base} key={f'已设置({len(emb_key)})' if emb_key else '缺失'}"
        f" EMBEDDING_PROVIDER={env.get('EMBEDDING_PROVIDER', 'api')}"
        f" EMBEDDING_MODEL={env.get('EMBEDDING_MODEL', '-')}",
        flush=True,
    )
    print("", flush=True)

    report = ProbeReport(chat_base=base, embedding_base=emb_base, key_present=bool(key), key_length=len(key))
    report.embedding_key_present = bool(emb_key)

    chat_models = args.chat_only or [m for m, _ in CHAT_CANDIDATES]
    chat_notes = dict(CHAT_CANDIDATES)
    chat_results = [probe_chat(base, key, m, args.timeout) for m in chat_models]
    for r in chat_results:
        if not r.ok and not r.detail:
            r.detail = chat_notes.get(r.model, "")
    report.results += chat_results
    _print_table("chat", chat_results)

    print("", flush=True)
    emb_models = args.emb_only or EMBEDDING_CANDIDATES
    if emb_key and emb_base:
        emb_results = [probe_embedding(emb_base, emb_key, m, args.timeout) for m in emb_models]
    else:
        # 缺 Key 时也要给出结论（否则读者会以为"没探"而不是"没配"）
        emb_results = [
            ProbeResult("embedding", m, False, "未配置", "NO_KEY", "EMBEDDING_API_KEY / BASE_URL 缺失")
            for m in emb_models
        ]
    report.results += emb_results
    _print_table("embedding", emb_results)

    usable_chat = report.usable("chat")
    usable_emb = report.usable("embedding")
    print("", flush=True)
    print("=== 结论 ===", flush=True)
    print(f"  可用会话模型：{', '.join(usable_chat) or '无'}", flush=True)
    print(f"  可用向量模型：{', '.join(usable_emb) or '无'}", flush=True)
    if usable_chat and usable_emb:
        hint = f"建议 .env：MODEL_NAME={usable_chat[0]} / MODEL_STRONG={usable_chat[0]}"
    elif usable_chat:
        hint = "向量侧需兜底：EMBEDDING_PROVIDER=local（本地模型无需 Key，且换向量后必须重建索引）"
    else:
        hint = "无可用会话模型：检查 Key 有效性 / 是否只剩付费模型 / 账号是否欠费"
    print(f"  建议：{hint}", flush=True)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        payload["suggestion"] = hint
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结论已写入：{out}", flush=True)

    if not usable_chat:
        return 20
    if not usable_emb:
        return 10
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
