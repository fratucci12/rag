import sys
import types
import importlib


def _install_stub_openai_with_tool_args(args_json: str):
    mod = types.ModuleType("openai")
    class _DummyChoice:
        def __init__(self, args):
            self.message = types.SimpleNamespace(
                tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(arguments=args))]
            )
    class _DummyCompletions:
        def __init__(self, args):
            self._args = args
        def create(self, **kwargs):
            return types.SimpleNamespace(choices=[_DummyChoice(self._args)])
    class _DummyChat:
        def __init__(self, args):
            self.completions = _DummyCompletions(args)
    class OpenAI:
        def __init__(self, *a, **k):
            self.chat = _DummyChat(args_json)
    sys.modules["openai"] = mod
    mod.OpenAI = OpenAI


def _reload_agent():
    if "rag_app.agent" in sys.modules:
        del sys.modules["rag_app.agent"]
    return importlib.import_module("rag_app.agent")


def test_query_planner_parses_tool_call_arguments():
    args_json = (
        '{"semantic_query": "preço de cadeiras", "ano": 2024, "estado_sigla": "SP"}'
    )
    _install_stub_openai_with_tool_args(args_json)
    agent = _reload_agent()

    qp = agent.QueryPlanner(model="gpt-4o-mini")
    out = qp.analyze_query("Qual o preço de cadeiras no estado de SP em 2024?")
    assert out["semantic_query"] == "preço de cadeiras"
    assert out["filters"]["ano"] == 2024
    assert out["filters"]["estado_sigla"] == "SP"

