"""perry/models.py is the only module that knows the model providers: no other module reads
their settings from the environment. (Ruff's TID251 bans their imports; this covers the
settings, which no import rule sees.)"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROVIDER_SETTING = re.compile(
    r"""(getenv|environ\.get|environ\[)\(?\s*["'](PERRY_INET_QWEN\w*|OPENAI_API_KEY|GOOGLE_API_KEY)"""
)


def test_only_models_reads_provider_settings():
    readers = [
        f"{path.relative_to(ROOT)}: {match.group(0)}"
        for folder in ("perry", "evals", "scripts", "benchmark")
        for path in (ROOT / folder).rglob("*.py")
        if path != ROOT / "perry" / "models.py"
        for match in PROVIDER_SETTING.finditer(path.read_text(encoding="utf-8"))
    ]
    assert readers == [], "read these through perry/models.py instead"
