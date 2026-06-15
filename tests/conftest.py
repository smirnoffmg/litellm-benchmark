import json
import urllib.error
import urllib.request
from typing import Any

import pytest


@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
    except (urllib.error.URLError, OSError):
        pytest.skip("Ollama not reachable at http://localhost:11434")
    return "http://localhost:11434"


@pytest.fixture(scope="session")
def ollama_model(ollama_base_url: str) -> str:
    with urllib.request.urlopen(f"{ollama_base_url}/api/tags", timeout=5) as resp:
        data: Any = json.loads(resp.read())
    models: list[dict[str, Any]] = data.get("models", [])
    if not models:
        pytest.skip("No Ollama models installed")
    return str(models[0]["name"])
