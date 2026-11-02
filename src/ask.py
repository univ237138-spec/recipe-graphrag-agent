"""GraphRAG로 질문에 답한다: 질문 해석(LLM) → 그래프 조회(Cypher) → [추천이면 리랭킹] → 안전 검사 → 설명(LLM).

추천 질문은 그래프가 거른 안전한 후보 안에서 질문 유사도 + 다양성(MMR)으로 순서를 정한다 (src/rerank.py).
레시피 임베딩(data/vector_index.npz)이 없거나 질문 임베딩을 못 만들면 레시피 번호순. --no-rerank로 끌 수 있다.

사용법:
    python src/ask.py "우유랑 새우 알레르기가 있어. 반찬 3개 추천해줘" --reasoning-effort low
    python src/ask.py "…" --no-llm-answer          # 설명 문단 없이 그래프 결과만 (LLM은 질문 해석에만 사용)
    python src/ask.py --conditions '{"intent": "check_safety", "recipe_name": "석류 보쌈김치", "exclude_allergens": ["새우", "게"]}'
                                                    # 질문 해석도 건너뛰기 (API 키 없이 그래프만 확인)
    python src/ask.py                               # 질문을 여러 개 이어서 하기 (대화 모드)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402
import answer  # noqa: E402
import graph_search  # noqa: E402
import llm  # noqa: E402
import load_neo4j  # noqa: E402
import ontology  # noqa: E402
import query_parser  # noqa: E402
import recipe_graph  # noqa: E402
import rerank  # noqa: E402

DEFAULT_CONDITIONS = {"intent": "recommend", "recipe_name": "", "exclude_allergens": [], "include_allergens": [],
                      "include_categories": [],
                      "dish_type": "", "cooking_method": "", "max_kcal": None, "max_sodium_mg": None, "limit": 3}


class GraphRAG:
    def __init__(self, run, graph, recipes, model=None, reasoning_effort=None, reranker=None):
        self.run, self.graph, self.recipes = run, graph, recipes
        self.model, self.reasoning_effort = llm.model_from_env(model), reasoning_effort
        self.reranker = reranker
        self.workflow = agent.build(self)

    def conditions(self, question=None, given=None):
        if given is not None:
            return query_parser.validate({**DEFAULT_CONDITIONS, **given}, self.graph)
        return query_parser.parse(question, self.graph, self.model, self.reasoning_effort)

    def ask(self, question=None, conditions=None, explain=True):
        """단계별 흐름은 src/agent.py (LangGraph)."""
        state = self.workflow.invoke({"question": question, "given": conditions, "explain": explain})
        return {"conditions": state["conditions"], "result": state["result"], "paragraphs": state.get("paragraphs"),
                "safety_notes": state.get("safety_notes", []), "llm_error": state.get("llm_error")}


def show(output):
    print("[조건]", json.dumps(output["conditions"], ensure_ascii=False))
    if output["llm_error"]:
        print(f"[설명 생략] {output['llm_error']}")
    print(answer.render(output["result"], output["paragraphs"], output["safety_notes"]))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="레시피·알레르기 GraphRAG")
    parser.add_argument("question", nargs="?", help="질문 (없으면 대화 모드)")
    parser.add_argument("--conditions", help="질문 해석을 건너뛰고 조건 JSON을 직접 지정")
    parser.add_argument("--no-llm-answer", action="store_true", help="설명 문단 없이 그래프 결과만")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--no-rerank", action="store_true", help="추천을 리랭킹 없이 레시피 번호순으로")
    args = parser.parse_args()

    graph, recipes = ontology.load(), recipe_graph.load_recipes()
    given = json.loads(args.conditions) if args.conditions else None
    driver, database = load_neo4j.connect()
    with driver, driver.session(database=database) as session:
        rag = GraphRAG(graph_search.session_runner(session), graph, recipes, args.model, args.reasoning_effort,
                       None if args.no_rerank else rerank.load(recipes=recipes))

        def answer_one(question):
            try:
                output = rag.ask(question, given, explain=not args.no_llm_answer)
            except llm.LLMError as error:
                print(f"오류: {error}")
                return False
            if args.json:
                print(json.dumps(output, ensure_ascii=False, indent=1, default=str))
            else:
                show(output)
            return True

        if args.question or given:
            sys.exit(0 if answer_one(args.question) else 1)
        while True:  # 대화 모드
            try:
                question = input("\n질문 (끝내려면 그냥 Enter): ").strip()
            except EOFError:
                break
            if not question:
                break
            answer_one(question)


if __name__ == "__main__":
    main()
