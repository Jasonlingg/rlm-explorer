"""Tests for the persistent REPL."""

import pytest

from src.env.repl import LocalREPL, PersistentREPL


@pytest.fixture
def repl() -> LocalREPL:
    """Create a local REPL for testing."""
    r = LocalREPL(corpus_path="data/corpus")
    r.start_session()
    yield r
    r.kill_session()


def test_state_persistence(repl: LocalREPL) -> None:
    """Variables from step 1 should be available in step 2."""
    repl.execute("x = 42")
    output = repl.execute("print(x)")
    assert "42" in output


def test_function_persistence(repl: LocalREPL) -> None:
    """Functions defined in step 1 should be callable in step 2."""
    repl.execute("def double(n): return n * 2")
    output = repl.execute("print(double(21))")
    assert "42" in output


def test_tools_available(repl: LocalREPL) -> None:
    """Tool preamble should make search/read available."""
    output = repl.execute("print(type(search))")
    assert "function" in output


def test_search_tool(repl: LocalREPL) -> None:
    """search() should return results from the corpus."""
    output = repl.execute('results = search("revenue"); print(len(results))')
    # Should have at least 1 result
    assert any(c.isdigit() and int(c) > 0 for c in output.split() if c.isdigit())


def test_read_tool(repl: LocalREPL) -> None:
    """read() should return document text."""
    output = repl.execute('text = read("apex_corp_2024_financial"); print("Apex" in text)')
    assert "True" in output


def test_timeout(repl: LocalREPL) -> None:
    """Long-running code should timeout."""
    output = repl.execute("import time; time.sleep(10)", timeout=2)
    assert "timeout" in output.lower() or "ERROR" in output


def test_session_cleanup() -> None:
    """kill_session should clean up without errors."""
    r = LocalREPL(corpus_path="data/corpus")
    r.start_session()
    r.execute("x = 1")
    r.kill_session()
    # Should not raise
    r.kill_session()


def test_auto_select_local() -> None:
    """PersistentREPL should fall back to local when Docker unavailable."""
    repl = PersistentREPL(use_docker=False, corpus_path="data/corpus")
    repl.start_session()
    output = repl.execute("print(1 + 1)")
    assert "2" in output
    repl.kill_session()
