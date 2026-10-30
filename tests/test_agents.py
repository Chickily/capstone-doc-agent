import json

from src.agents import AgentOrchestrator
from src.guardrails import NO_EVIDENCE_MSG
from src.ingestion import process_upload
from src.llm import LLM


class FakeLLM:
    """Returns scripted replies so the agent loop can be tested without any API."""

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, system, messages, max_tokens=1024):
        return self.replies.pop(0)


class FakeStore:
    def search(self, query, k):
        return []


def _tables():
    data = b"region,revenue\nTokyo,100\nOsaka,50\nTokyo,20\n"
    return process_upload("sales.csv", data)[1]


def test_agent_uses_table_tool_then_validates():
    llm = FakeLLM([
        json.dumps({"intent": "sum", "needs_documents": True}),
        json.dumps({"action": "tool", "name": "analyze_table",
                    "args": {"table": "sales.csv", "operation": "aggregate",
                             "column": "revenue", "agg": "sum"}}),
        json.dumps({"action": "final", "answer": "Total revenue is 170 [sales.csv]"}),
        json.dumps({"grounded": True, "unsupported_claims": [], "final_answer": "Total revenue is 170 [sales.csv]"}),
    ])
    result = AgentOrchestrator(llm, FakeStore(), _tables()).run("total revenue?", [])
    assert "170" in result["answer"]
    assert any(c.get("tool") == "analyze_table" for c in result["tool_calls"])
    assert "sales.csv (table analysis)" in result["sources"]


def test_agent_refuses_when_no_evidence():
    llm = FakeLLM([
        json.dumps({"needs_documents": True}),
        json.dumps({"action": "final", "answer": "The capital is Paris."}),   # never used a tool
    ])
    result = AgentOrchestrator(llm, FakeStore(), {}).run("capital of France?", [])
    assert result["answer"] == NO_EVIDENCE_MSG


def test_llm_rejects_unknown_provider_and_missing_key():
    for args in [("", "m", "gemini"), ("k", "m", "nope")]:
        try:
            LLM(*args)
            assert False, "should have raised"
        except ValueError:
            pass


def test_gemini_client_builds_without_network():
    llm = LLM("fake-key", "gemini-flash-latest", "gemini")
    assert llm.provider == "gemini"


class HitStore:
    """Pretends the vector DB found a relevant passage."""

    def search(self, query, k):
        return [{"text": "Employees may work remotely up to two days per week.",
                 "metadata": {"source": "doc.pdf", "location": "page 2"}, "distance": 0.2}]


def test_agent_answers_even_if_model_skips_tools():
    # Researcher answers immediately with no tool call; the automatic retrieval still supplies evidence
    llm = FakeLLM([
        json.dumps({"needs_documents": True, "search_queries": ["remote work days"]}),
        json.dumps({"action": "final", "answer": "Two days per week [doc.pdf, page 2]"}),
        json.dumps({"grounded": True, "unsupported_claims": [], "final_answer": "Two days per week [doc.pdf, page 2]"}),
    ])
    result = AgentOrchestrator(llm, HitStore(), {}).run("remote days?", [])
    assert "Two days" in result["answer"]
    assert "doc.pdf, page 2" in result["sources"]
    assert result["tool_calls"][0]["tool"] == "search_documents (automatic)"


def test_agent_accepts_tool_name_as_action():
    llm = FakeLLM([
        json.dumps({"needs_documents": True}),
        json.dumps({"action": "search_documents", "args": {"query": "remote"}}),   # non-standard format
        json.dumps({"action": "final", "answer": "Two days"}),
        json.dumps({"final_answer": "Two days"}),
    ])
    result = AgentOrchestrator(llm, HitStore(), {}).run("remote days?", [])
    assert result["answer"] == "Two days"
    assert any(c.get("tool") == "search_documents" for c in result["tool_calls"])


def test_agent_handles_plain_text_reply():
    llm = FakeLLM([
        json.dumps({"needs_documents": True}),
        "Employees can work remotely two days per week.",          # not JSON at all
        json.dumps({"final_answer": "Two days per week."}),
    ])
    result = AgentOrchestrator(llm, HitStore(), {}).run("remote days?", [])
    assert result["answer"] == "Two days per week."
