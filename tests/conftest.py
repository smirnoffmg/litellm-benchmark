import json
import os
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


def _capabilities(base_url: str, model: str) -> list[str]:
    request = urllib.request.Request(
        f"{base_url}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as resp:
        data: Any = json.loads(resp.read())
    return list(data.get("capabilities", []))


@pytest.fixture(scope="session")
def ollama_model(ollama_base_url: str) -> str:
    if override := os.environ.get("OLLAMA_TEST_MODEL"):
        return override
    with urllib.request.urlopen(f"{ollama_base_url}/api/tags", timeout=5) as resp:
        data: Any = json.loads(resp.read())
    names = [str(m["name"]) for m in data.get("models", [])]
    # Embedding models reject chat; thinking models may spend the whole reply on
    # hidden reasoning and stream no content, which would fail the TTFT assertion.
    for name in names:
        caps = _capabilities(ollama_base_url, name)
        if "completion" in caps and "thinking" not in caps:
            return name
    pytest.skip("No Ollama chat model installed (need 'completion' without 'thinking')")
