import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_loaders_registered():
    from ingest import LOADERS
    assert {".pdf", ".txt", ".md"} <= set(LOADERS)


def test_prompt_requires_context_grounding():
    from rag import PROMPT
    assert "ONLY the context" in PROMPT
    assert "{context}" in PROMPT and "{question}" in PROMPT
