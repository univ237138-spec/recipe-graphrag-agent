"""레시피·알레르기 GraphRAG API (FastAPI). 질문 하나를 LangGraph 워크플로우(src/agent.py)로 처리한다.

    uvicorn api:app --reload        → http://localhost:8000/docs

    POST /ask {"question": "우유 알레르기가 있어. 반찬 3개 추천해줘"}
    POST /ask {"conditions": {"intent": "check_safety", "recipe_name": "석류 보쌈김치", "exclude_allergens": ["새우"]},
               "explain": false}                       # 질문 해석·설명 없이 그래프만 (API 키 불필요)

환경변수: OPENAI_API_KEY, OPENAI_MODEL (기본 gpt-5-mini).
NEO4J_PASSWORD(+ NEO4J_URI 등)가 있으면 Neo4j, 없으면 메모리 그래프(local_graph, Neo4j와 결과 일치 검증됨).
응답의 metrics: 지연(ms), LLM 호출 수, 입력·출력 토큰.
"""

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import ask  # noqa: E402
import llm  # noqa: E402
import load_neo4j  # noqa: E402
import local_graph  # noqa: E402
import recipe_graph  # noqa: E402
import rerank  # noqa: E402


class Conditions(BaseModel):
    intent: str
    recipe_name: str = ""
    exclude_allergens: list[str] = []
    include_allergens: list[str] = []
    include_categories: list[str] = []
    dish_type: str = ""
    cooking_method: str = ""
    max_kcal: float | None = None
    max_sodium_mg: float | None = None
    limit: int = 3


class AskRequest(BaseModel):
    question: str | None = Field(None, max_length=500)  # LLM 비용 상한
    conditions: Conditions | None = None  # 주면 질문 해석(LLM)을 건너뛴다
    explain: bool = True


def make_runner(graph, recipes, mapping, mentions):
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        return local_graph.LocalRunner(graph, recipes, mapping, mentions), None
    driver, database = load_neo4j.open_driver(password)

    def run(query, **params):  # 요청이 동시에 들어오므로 쿼리마다 세션을 연다 (세션은 스레드 간 공유 불가)
        with driver.session(database=database) as session:
            return [record.data() for record in session.run(query, **params)]
    return run, driver


@asynccontextmanager
async def lifespan(app):
    graph, recipes, mapping, mentions = recipe_graph.load_all()
    run, driver = make_runner(graph, recipes, mapping, mentions)
    app.state.rag = ask.GraphRAG(run, graph, recipes, os.environ.get("OPENAI_MODEL") or "gpt-5-mini",
                                 reranker=rerank.load(recipes=recipes))
    yield
    if driver:
        driver.close()


app = FastAPI(title="레시피·알레르기 GraphRAG", lifespan=lifespan)


def answer_with_metrics(rag, question, given, explain):
    calls = []
    llm.USAGE.set(calls)  # 이 요청의 LLM 호출만 모인다 (to_thread가 컨텍스트를 복사해 요청끼리 섞이지 않음)
    start = time.perf_counter()
    if given is not None:
        try:
            rag.conditions(given=given)
        except llm.LLMError as exc:  # 온톨로지에 없는 알레르기·분류 등 입력 오류
            raise HTTPException(422, str(exc)) from None
    try:
        output = rag.ask(question, given, explain)
    except llm.LLMError as exc:  # 질문 해석 실패 (키 없음·API 오류·해석 불가)
        raise HTTPException(502, str(exc)) from None
    return {**output, "metrics": {"latency_ms": round((time.perf_counter() - start) * 1000),
                                  "llm_calls": len(calls),
                                  "input_tokens": sum(c["input_tokens"] for c in calls),
                                  "output_tokens": sum(c["output_tokens"] for c in calls)}}


@app.post("/ask")
async def ask_endpoint(body: AskRequest):
    if not (body.question and body.question.strip()) and body.conditions is None:
        raise HTTPException(422, "question 또는 conditions가 필요합니다.")
    given = body.conditions.model_dump(exclude_unset=True) if body.conditions else None
    # 그래프 조회·LLM 호출이 동기 코드라 스레드에서 돌려 이벤트 루프를 막지 않는다
    return await asyncio.to_thread(answer_with_metrics, app.state.rag, body.question, given, body.explain)
