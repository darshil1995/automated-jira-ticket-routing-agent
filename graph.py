"""
LangGraph Multi-Agent Graph — JIRA Workflow Pipeline Orchestrator.

This module defines and compiles the stateful directed graph that connects
all three agents into a single coherent pipeline. It is the central nervous
system of the entire project.

What is LangGraph?
    LangGraph models an agent pipeline as a directed graph where:
        - Nodes  = agents or processing steps (functions that read/write state)
        - Edges  = the flow of control between nodes
        - State  = a shared typed dictionary that every node can read from
                   and write to

    Unlike a simple function chain (A calls B calls C), LangGraph:
        - Maintains a persistent shared state across all nodes
        - Supports conditional routing (e.g. skip Resolution if already resolved)
        - Provides built-in error handling and retry logic per node
        - Makes the pipeline inspectable — you can see exactly what state
          looked like at every step, which is critical for debugging LLM systems

Graph structure:
    [START]
       │
       ▼
   [triage]          ← Agent A: classify issue, check logs
       │
       ▼
   [resolution]      ← Agent B: RAG search + generate ticket draft
       │
       ▼
       [qa]          ← Agent C: privacy + compliance review
       │
       ▼
     [END]

State flow:
    Input:  { "issue": "raw issue string" }
    After triage:     adds "triage" key
    After resolution: adds "resolution" key
    After qa:         adds "qa" key (contains final approved ticket)

Dependencies:
    - langgraph: StateGraph, START, END primitives
    - typing: TypedDict for typed state schema
    - agents.*: the three agent node functions
"""

import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from agents.triage_agent import run_triage
from agents.resolution_agent import run_resolution
from agents.qa_agent import run_qa
import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    Shared state dictionary passed between all nodes in the graph.

    TypedDict gives us type hints and IDE autocompletion while keeping
    the state a plain Python dict under the hood — which is what LangGraph
    expects. Every node receives the full state and returns a fragment
    (only the keys it updates) which LangGraph merges back in.

    total=False means all keys are optional — the graph starts with only
    "issue" populated and the remaining keys are added as agents run.
    Without total=False, TypedDict would require all keys to be present
    at initialisation, which would break the incremental state pattern.

    Attributes:
        issue:      The raw incoming issue string. Set at graph entry.
                    Never modified by any agent — it's the immutable input.

        triage:     Structured output from the Triage Agent (Agent A).
                    Set after the triage node runs. Contains category,
                    priority, log_summary, recommended_action, confidence.

        resolution: Structured output from the Resolution Agent (Agent B).
                    Set after the resolution node runs. Contains ticket_title,
                    ticket_description, resolution_steps, affected_systems,
                    escalation_path, estimated_resolution_time.

        qa:         Structured output from the QA Agent (Agent C).
                    Set after the qa node runs. Contains approved,
                    violations, risk_level, final_ticket, qa_notes.
                    This is the terminal output of the pipeline.
    """

    issue: str
    triage: dict[str, Any]
    resolution: dict[str, Any]
    qa: dict[str, Any]


# ---------------------------------------------------------------------------
# Node Wrappers
# ---------------------------------------------------------------------------
# Each wrapper adds structured logging around the agent call so every
# node transition appears clearly in CloudWatch with timing context.
# The actual agent logic lives in the agents/ modules — these are
# thin observability wrappers only.

def triage_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the Triage Agent.

    Logs node entry and exit with state key counts so you can
    trace the pipeline execution in CloudWatch at a glance.

    Args:
        state: Current graph state. Must contain "issue".

    Returns:
        Dict fragment with "triage" key for LangGraph to merge into state.
    """
    logger.info(
        "Graph: entering triage_node | state_keys=%s",
        list(state.keys())
    )

    result = run_triage(state)

    logger.info(
        "Graph: exiting triage_node | added_keys=%s",
        list(result.keys())
    )
    return result


def resolution_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the Resolution Agent.

    Args:
        state: Current graph state. Must contain "triage".

    Returns:
        Dict fragment with "resolution" key for LangGraph to merge into state.
    """
    logger.info(
        "Graph: entering resolution_node | state_keys=%s",
        list(state.keys())
    )

    result = run_resolution(state)

    logger.info(
        "Graph: exiting resolution_node | added_keys=%s",
        list(result.keys())
    )
    return result


def qa_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the QA Agent.

    Args:
        state: Current graph state. Must contain "resolution".

    Returns:
        Dict fragment with "qa" key for LangGraph to merge into state.
    """
    logger.info(
        "Graph: entering qa_node | state_keys=%s",
        list(state.keys())
    )

    result = run_qa(state)

    logger.info(
        "Graph: exiting qa_node | added_keys=%s | approved=%s",
        list(result.keys()),
        result.get("qa", {}).get("approved")
    )
    return result


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Constructs and compiles the multi-agent LangGraph pipeline.

    Build steps:
        1. Create a StateGraph typed to AgentState
        2. Register each agent as a named node
        3. Define edges (the execution order)
        4. Compile into an executable graph object

    Why compile?
        Compilation validates the graph structure — it catches missing
        nodes, disconnected edges, and unreachable states before runtime.
        The compiled graph is also serialisable, which means LangGraph
        can checkpoint state mid-pipeline for long-running workflows.

    Returns:
        A compiled LangGraph StateGraph ready to invoke.
    """
    logger.info("Building LangGraph pipeline")

    # Step 1: Create graph with our typed state schema
    graph = StateGraph(AgentState)

    # Step 2: Register nodes
    # The string name ("triage", "resolution", "qa") is what appears
    # in LangGraph traces and CloudWatch logs — choose names that make
    # the pipeline readable at a glance
    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("qa", qa_node)

    # Step 3: Define edges — the execution order
    # START → triage → resolution → qa → END
    # This is a linear pipeline with no branching.
    #
    # To add conditional routing later (e.g. skip resolution for P4 issues):
    #   graph.add_conditional_edges(
    #       "triage",
    #       lambda state: "end" if state["triage"]["priority"] == "P4" else "resolution",
    #       { "resolution": "resolution", "end": END }
    #   )
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "qa")
    graph.add_edge("qa", END)

    # Step 4: Compile
    compiled = graph.compile()

    logger.info("LangGraph pipeline compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# Module-level compiled graph
# ---------------------------------------------------------------------------

# Compiled once at import time and reused across Lambda warm starts.
# Compilation is moderately expensive — doing it inside the Lambda handler
# would add ~200ms to every cold start. Module-level keeps it to once.
pipeline = build_graph()


def run_pipeline(issue: str) -> dict[str, Any]:
    """
    Public entry point for the full JIRA agent pipeline.

    Accepts a raw issue string, runs it through all three agents,
    and returns the complete final state including triage classification,
    resolution draft, and QA approval decision.

    This is what lambda_handler.py calls — it knows nothing about
    LangGraph internals, it just calls run_pipeline() and gets back
    a dict.

    Args:
        issue: Raw incoming issue text. Can be a monitoring alert,
               a Slack message, a user bug report, or a system event.

    Returns:
        Complete AgentState dict containing all four keys:
            - issue:      the original input (unchanged)
            - triage:     Agent A classification result
            - resolution: Agent B ticket draft
            - qa:         Agent C compliance review + final ticket

    Raises:
        ValueError: Propagated from any agent if LLM returns invalid JSON.
        RuntimeError: If the graph encounters an unrecoverable state error.
    """
    logger.info(
        "run_pipeline() called | issue_preview='%s'",
        issue[:100]
    )

    # Invoke the compiled graph with the initial state
    # LangGraph runs each node in order, merging results into state
    final_state = pipeline.invoke({"issue": issue})

    logger.info(
        "Pipeline complete | approved=%s | priority=%s | violations=%d",
        final_state.get("qa", {}).get("approved"),
        final_state.get("triage", {}).get("priority"),
        final_state.get("qa", {}).get("violation_count", 0)
    )

    return final_state