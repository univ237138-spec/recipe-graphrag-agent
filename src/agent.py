"""GraphRAG.ask()를 LangGraph 워크플로우로 실행한다.

    parse → retrieve ─┬─ (추천) → [rerank] → select → safety ─┬─ explain → verify ─┬─ 끝
                      │                        ↑  (빠진 요리가 있으면 1번 재조회) ┘    ↑  (검증 실패면 2번 재생성) ┘
                      ├─ (레시피 없음 / 설명 끔) → 끝
                      └─ (그 밖의 질문) → explain

안전 판단은 그래프(Cypher)만 한다. LLM은 질문 해석·질문 확장·설명에만 쓴다.
설명이 끝까지 검증을 통과하지 못하면 설명 없이 그래프 결과와 llm_error를 돌려준다.

    python src/agent.py      # 워크플로우 그림(Mermaid) 출력
"""

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer  # noqa: E402
import graph_search  # noqa: E402
import llm  # noqa: E402

MAX_REQUERIES = 1        # 안전 검사로 요리가 빠졌을 때 다시 고르는 횟수
MAX_EXPLAIN_RETRIES = 2  # 근거 검증이 실패했을 때 설명을 다시 요청하는 횟수


class State(TypedDict, total=False):
    question: str | None
    given: dict | None          # 질문 해석을 건너뛸 때 주는 조건
    explain: bool
    conditions: dict
    candidates: list            # 추천: 조건에 맞는 요리 전부 (안전 필터 전)
    ranker: Any                 # 추천: 리랭킹 함수 (없으면 레시피 번호순)
    rounds: int                 # 추천: 고르기를 몇 번 했는지 (재조회 포함)
    result: dict
    safety_notes: list
    evidence: dict
    reply: dict | None
    attempts: int               # 설명 요청 횟수
    paragraphs: list | None
    llm_error: str | None


def build(rag):
    """rag(ask.GraphRAG)의 조회 함수·온톨로지·모델 설정으로 워크플로우를 만든다."""

    def parse(state):
        return {"conditions": rag.conditions(state.get("question"), state.get("given"))}

    def retrieve(state):
        cond = state["conditions"]
        if cond["intent"] == "recommend":
            return {"candidates": graph_search.recommend_candidates(rag.run, cond)}
        return {"result": graph_search.search(rag.run, cond, rag.recipes)}

    def rerank(state):
        cond = state["conditions"]
        return {"ranker": rag.reranker.for_question(state["question"], cond["exclude_allergens"], rag.model)}

    def select(state):
        cond = state["conditions"]
        limit = cond["limit"] + len(state.get("safety_notes", []))  # 재조회면 빠진 만큼 더 고른다
        result = graph_search.recommend(rag.run, cond, state["candidates"], state.get("ranker"), limit)
        return {"result": result, "rounds": state.get("rounds", 0) + 1}

    def safety(state):
        return {"safety_notes": answer.safety_check(rag.run, state["result"])}

    def explain(state):
        result, cond = state["result"], state["conditions"]
        evidence = answer.build_evidence(result)
        request = answer.build_request(state.get("question") or json.dumps(cond, ensure_ascii=False),
                                       result, evidence, rag.model, rag.reasoning_effort)
        if state.get("llm_error"):  # 재생성: 직전 검증 실패 이유를 알려 준다
            request["input"].append({"role": "user", "content":
                                     f"직전 답변이 검증에 실패했습니다: {state['llm_error']} "
                                     "모든 문단에 evidence에 있는 ID만 근거로 붙이세요."})
        attempts = state.get("attempts", 0) + 1
        try:
            return {"evidence": evidence, "reply": llm.call(request), "attempts": attempts, "llm_error": None}
        except llm.LLMError as exc:
            return {"evidence": evidence, "reply": None, "attempts": attempts, "llm_error": str(exc)}

    def verify(state):
        try:
            return {"paragraphs": answer.validate_paragraphs(state["reply"], state["evidence"]), "llm_error": None}
        except llm.LLMError as exc:
            return {"paragraphs": None, "llm_error": str(exc)}

    def explain_or_end(state):
        return "explain" if state.get("explain") else END

    def after_retrieve(state):
        if state["conditions"]["intent"] == "recommend":
            return "rerank" if rag.reranker and state.get("question") else "select"
        if state["result"].get("not_found") is not None:
            return END
        return explain_or_end(state)

    def after_safety(state):
        short = len(state["result"]["recommended"]) < state["conditions"]["limit"]
        if state["safety_notes"] and short and state["rounds"] <= MAX_REQUERIES:
            return "select"
        return explain_or_end(state)

    def after_explain(state):
        return "verify" if state["reply"] is not None else END  # API 오류는 재시도하지 않는다

    def after_verify(state):
        if state["paragraphs"] is None and state["attempts"] <= MAX_EXPLAIN_RETRIES:
            return "explain"
        return END

    flow = StateGraph(State)
    for node in (parse, retrieve, rerank, select, safety, explain, verify):
        flow.add_node(node.__name__, node)
    flow.add_edge(START, "parse")
    flow.add_edge("parse", "retrieve")
    flow.add_conditional_edges("retrieve", after_retrieve, ["rerank", "select", "explain", END])
    flow.add_edge("rerank", "select")
    flow.add_edge("select", "safety")
    flow.add_conditional_edges("safety", after_safety, ["select", "explain", END])
    flow.add_conditional_edges("explain", after_explain, ["verify", END])
    flow.add_conditional_edges("verify", after_verify, ["explain", END])
    return flow.compile()


if __name__ == "__main__":
    class _Stub:  # 그림만 그릴 때는 실제 데이터가 필요 없다
        run = graph = recipes = model = reasoning_effort = reranker = None
    print(build(_Stub()).get_graph().draw_mermaid())
