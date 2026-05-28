import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.triage_agent import run_triage
from agents.resolution_agent import run_resolution
from agents.qa_agent import run_qa
import config

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """
    Shared state passed between every node in the pipeline.

    total=False allows incremental population — the graph starts with
    only 'issue' set, and each agent appends its own key as it runs.
    """
    issue: str
    triage: dict[str, Any]
    resolution: dict[str, Any]
    qa: dict[str, Any]


def triage_node(state: AgentState) -> dict:
    logger.info("Graph: entering triage_node | state_keys=%s", list(state.keys()))
    result = run_triage(state)
    logger.info("Graph: exiting triage_node | added_keys=%s", list(result.keys()))
    return result


def resolution_node(state: AgentState) -> dict:
    logger.info("Graph: entering resolution_node | state_keys=%s", list(state.keys()))
    result = run_resolution(state)
    logger.info("Graph: exiting resolution_node | added_keys=%s", list(result.keys()))
    return result


def qa_node(state: AgentState) -> dict:
    logger.info("Graph: entering qa_node | state_keys=%s", list(state.keys()))
    result = run_qa(state)
    logger.info(
        "Graph: exiting qa_node | added_keys=%s | approved=%s",
        list(result.keys()),
        result.get("qa", {}).get("approved")
    )
    return result


def build_graph() -> StateGraph:
    """
    Compiles the three-node LangGraph pipeline.

    Compilation validates the graph structure at build time — catching
    disconnected edges and missing nodes before any invocation occurs.
    """
    logger.info("Building LangGraph pipeline")

    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("qa", qa_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "qa")
    graph.add_edge("qa", END)

    # To add conditional routing in future (e.g. skip resolution for P4):
    # graph.add_conditional_edges(
    #     "triage",
    #     lambda state: "end" if state["triage"]["priority"] == "P4" else "resolution",
    #     {"resolution": "resolution", "end": END}
    # )

    compiled = graph.compile()
    logger.info("LangGraph pipeline compiled successfully")
    return compiled


# Compiled once at module load and reused across Lambda warm starts.
# Compiling inside handler() would add ~200ms to every invocation.
pipeline = build_graph()


def run_pipeline(issue: str) -> dict[str, Any]:
    """
    Runs the full three-agent pipeline and returns the complete final state.

    Raises:
        ValueError: Propagated from any agent that receives invalid LLM JSON.
        RuntimeError: If the graph encounters an unrecoverable state error.
    """
    logger.info("run_pipeline() called | issue_preview='%s'", issue[:100])

    final_state = pipeline.invoke({"issue": issue})

    logger.info(
        "Pipeline complete | approved=%s | priority=%s | violations=%d",
        final_state.get("qa", {}).get("approved"),
        final_state.get("triage", {}).get("priority"),
        final_state.get("qa", {}).get("violation_count", 0)
    )

    return final_state