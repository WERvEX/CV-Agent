from __future__ import annotations

import tomllib
from pathlib import Path


def test_cli_scripts_include_underscore_and_compact_alias() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = data["project"]["scripts"]

    assert scripts["cv_agent"] == "cv_agent.cli.main:cli"
    assert scripts["cvagent"] == "cv_agent.cli.main:cli"
