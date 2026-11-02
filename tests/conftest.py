import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import ask  # noqa: E402
import local_graph  # noqa: E402
import recipe_graph  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return recipe_graph.load_all()  # graph, recipes, mapping, mentions


@pytest.fixture(scope="session")
def runner(data):
    return local_graph.LocalRunner(*data)


@pytest.fixture
def rag(data, runner):
    graph, recipes, _, _ = data
    return ask.GraphRAG(runner, graph, recipes, model="fake-model")
