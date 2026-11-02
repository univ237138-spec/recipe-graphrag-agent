"""FastAPI POST /ask 테스트 (API 키·Neo4j 없이)."""

import json

import pytest
from fastapi.testclient import TestClient

import api
import llm

SIDE = {"intent": "recommend", "exclude_allergens": ["우유"], "dish_type": "반찬", "limit": 3}


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.delenv("NEO4J_PASSWORD", raising=False)  # 메모리 그래프로
    with TestClient(api.app) as c:
        yield c
    mp.undo()


def test_graph_only_request(client):
    response = client.post("/ask", json={"conditions": SIDE, "explain": False})
    assert response.status_code == 200
    body = response.json()
    assert len(body["result"]["recommended"]) == 3
    assert body["metrics"]["llm_calls"] == 0 and body["metrics"]["latency_ms"] >= 0


def test_metrics_count_tokens(client, monkeypatch):
    def fake_post(url, payload, timeout=120):
        evidence = json.loads(payload["input"][0]["content"])["evidence"]
        text = json.dumps({"paragraphs": [{"text": "설명", "evidence_ids": [next(iter(evidence))]}]})
        response = {"status": "completed", "usage": {"input_tokens": 100, "output_tokens": 20},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        llm.record_usage(payload["model"], response)
        return response
    monkeypatch.setattr(llm, "post", fake_post)
    body = client.post("/ask", json={"conditions": SIDE}).json()
    assert body["paragraphs"] and body["llm_error"] is None
    assert body["metrics"] | {"latency_ms": 0} == {"latency_ms": 0, "llm_calls": 1, "input_tokens": 100, "output_tokens": 20}


@pytest.mark.parametrize("payload", [
    {},                                                                   # 질문도 조건도 없음
    {"question": "   "},
    {"question": "가" * 501},                                             # 너무 긴 질문
    {"conditions": {"intent": "recommend", "exclude_allergens": "우유"}},  # 타입 오류
    {"conditions": {"intent": "recommend", "exclude_allergens": ["없는알레르기"]}},
    {"conditions": {"intent": "아무거나"}},
])
def test_bad_input_is_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_question_without_key_is_502(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/ask", json={"question": "우유 알레르기가 있어. 반찬 추천해줘"})
    assert response.status_code == 502 and "OPENAI_API_KEY" in response.json()["detail"]
