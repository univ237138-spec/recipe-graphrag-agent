"""LangGraph 워크플로우 회귀 테스트 (API 키·Neo4j 없이, 메모리 그래프 + 가짜 LLM)."""

import pytest

import graph_search
import llm

SIDE = {"intent": "recommend", "exclude_allergens": ["우유", "새우"], "dish_type": "반찬", "limit": 3}


# ---------------------------------------------------------------- 그래프 결과 (기존 동작 그대로)

def test_recommend_matches_known_result(rag):
    result = rag.ask(conditions=SIDE, explain=False)["result"]
    assert [r["name"] for r in result["recommended"]] == ["백일송이버섯 볶음", "땅콩소스 버섯가지나물", "부추 콩가루 찜"]
    assert (result["allowed"], result["excluded_count"]) == (406, 168)


@pytest.mark.parametrize("allergens", [["우유"], ["밀"], ["대두", "알류"], ["게", "새우", "조개류"]])
def test_recommend_never_contains_excluded_allergen(rag, runner, allergens):
    result = rag.ask(conditions={"intent": "recommend", "exclude_allergens": allergens, "limit": 10},
                     explain=False)["result"]
    assert result["recommended"]
    for item in result["recommended"]:
        paths, _ = graph_search.recipe_evidence(runner, item["id"], allergens)
        assert not [p for p in paths if p["certainty"] == "확정"], item["name"]


def test_check_safety_verdict(rag):
    out = rag.ask(conditions={"intent": "check_safety", "recipe_name": "석류 보쌈김치",
                              "exclude_allergens": ["새우", "게"]}, explain=False)
    assert out["result"]["verdict"] == "unsafe"


def test_count(rag):
    assert rag.ask(conditions={"intent": "count", "include_allergens": ["땅콩"]}, explain=False)["result"]["count"] == 39


def test_unknown_recipe_ends_without_llm(rag, monkeypatch):
    monkeypatch.setattr(llm, "call", lambda request: pytest.fail("LLM을 부르면 안 된다"))
    out = rag.ask(conditions={"intent": "check_safety", "recipe_name": "없는요리이름"})
    assert out["result"]["not_found"] == "없는요리이름" and out["paragraphs"] is None


# ---------------------------------------------------------------- 재시도·재조회

def citing(evidence_id):
    return {"paragraphs": [{"text": "설명", "evidence_ids": [evidence_id]}]}


def first_evidence_id(request):
    import json
    return next(iter(json.loads(request["input"][0]["content"])["evidence"]))


def test_explain_retries_until_valid(rag, monkeypatch):
    calls = []

    def fake(request):
        calls.append(request)
        return citing("없는ID") if len(calls) < 3 else citing(first_evidence_id(request))
    monkeypatch.setattr(llm, "call", fake)
    out = rag.ask(conditions=SIDE)
    assert len(calls) == 3 and out["paragraphs"] and out["llm_error"] is None
    assert "직전 답변이 검증에 실패" in calls[1]["input"][-1]["content"]  # 실패 이유를 알려 준다


def test_explain_gives_up_after_two_retries(rag, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "call", lambda request: calls.append(request) or {"paragraphs": []})
    out = rag.ask(conditions=SIDE)
    assert len(calls) == 3  # 처음 1번 + 재생성 2번
    assert out["paragraphs"] is None and out["llm_error"] and out["result"]["recommended"]  # 그래프 결과는 남는다


def test_api_error_is_not_retried(rag, monkeypatch):
    calls = []

    def fail(request):
        calls.append(request)
        raise llm.LLMError("연결 실패")
    monkeypatch.setattr(llm, "call", fail)
    out = rag.ask(conditions=SIDE)
    assert len(calls) == 1 and out["llm_error"] == "연결 실패"


def test_requery_fills_recipe_dropped_by_safety_check(rag, monkeypatch):
    first = rag.ask(conditions=SIDE, explain=False)["result"]["recommended"][0]["id"]
    real = graph_search.recipe_evidence

    def evidence(run, recipe_id, allergens=None):  # 안전 검사가 첫 요리에서 확정 알레르기를 찾은 상황
        if recipe_id == first:
            return [{"allergen": "우유", "certainty": "확정"}], []
        return real(run, recipe_id, allergens)
    monkeypatch.setattr(graph_search, "recipe_evidence", evidence)
    out = rag.ask(conditions=SIDE, explain=False)
    ids = [r["id"] for r in out["result"]["recommended"]]
    assert len(ids) == 3 and first not in ids and len(out["safety_notes"]) == 1
