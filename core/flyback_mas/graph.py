import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import PowerSupplyState
from .nodes.requirements import requirements_agent_node
from .nodes.designer import circuit_designer_node
from .nodes.magnetic_advisor import magnetic_advisor_node
from .nodes.component_selector import component_selector_node
from .nodes.simulation import simulation_coordinator_node
from .nodes.validator import design_validator_node
from .nodes.correction import correction_agent_node
from .nodes.reporter import report_writer_node
from .nodes.skill_executor import skill_executor_node
from .nodes.memory_synthesizer import memory_synthesizer_node

try:
    from langgraph.store.postgres import PostgresStore
except Exception:
    PostgresStore = None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except Exception:
    PostgresSaver = None

def route_validation(state: PowerSupplyState):
    """Decides next step based on validation status."""
    msg_last = state.get("verification") or {}
    status = msg_last.get("status", "FAIL")
    # [HITL FIX] Check for explicit Human Intervention Request
    if status == "NEEDS_HUMAN_REVIEW":
        return END 
    
    # [HITL USER REQUEST]: Allow AUTO-RETRY on failure up to max_iterations
    # If explicit human intervention is requested, stopping is handled above.
        
    if status == "RETRY_REQUESTED":
        return "designer"
        
    iteration = state.get("iteration", 0)
    # [HITL FIX] Lower max auto-iterations to trigger human help sooner
    max_itr = state.get("max_iterations", 5) 
    
    if status == "PASS":
        return "correction"
    elif iteration >= max_itr:
        # Stop for human help if max iterations reached, BUT report the final state first
        print(f"⚠️ Max iterations ({max_itr}) reached. Generating Final Report with current (failed) state.")
        return "correction"
    elif status == "FAIL" or status == "TOPOLOGY_CHANGE_NEEDED":
        # [HITL MODE] Do not auto-iterate immediately.
        # The server layer provides an explicit post-validation checkpoint where user chooses:
        # accept current result / continue auto-iteration / manual adjustments.
        return END
    else:
        # Default loopback if status is something else or we want implicit
        return END

# Initialize Graph
workflow = StateGraph(PowerSupplyState)

# Add Nodes
workflow.add_node("requirements", requirements_agent_node)
workflow.add_node("designer", circuit_designer_node)
workflow.add_node("magnetics_advisor", magnetic_advisor_node)
workflow.add_node("selector", component_selector_node)
workflow.add_node("simulator", simulation_coordinator_node)
workflow.add_node("validator", design_validator_node)
workflow.add_node("correction", correction_agent_node)
workflow.add_node("memory_synthesizer", memory_synthesizer_node)
workflow.add_node("reporter", report_writer_node)
workflow.add_node("skill_executor", skill_executor_node)

def route_requirements(state: PowerSupplyState):
    """Route based on whether the request is chitchat or a design task."""
    if state.get("error_log"):
        return END

    request_profile = state.get("request_profile") or {}
    if request_profile.get("experiment_intent") == "corner_rerun":
        return "designer"

    # Check for Active Skill
    if state.get("active_skill"):
        return "skill_executor"

    specs = state.get("specifications") or {}
    if specs.get("is_chitchat"):
        return END
    return "designer"

# Add Edges
workflow.set_entry_point("requirements")
# Replace direct edge with conditional
workflow.add_conditional_edges(
    "requirements",
    route_requirements,
    {
        "designer": "designer",
        "skill_executor": "skill_executor",
        END: END
    }
)

workflow.add_edge("skill_executor", END)

workflow.add_edge("designer", "magnetics_advisor")
workflow.add_edge("magnetics_advisor", "selector")
workflow.add_edge("selector", "simulator")

workflow.add_edge("simulator", "validator")

# Add Conditional Edges
workflow.add_conditional_edges(
    "validator",
    route_validation,
    {
        "correction": "correction",
        "designer": "designer",
        END: END
    }
)

workflow.add_edge("reporter", END)
workflow.add_edge("correction", "memory_synthesizer")
workflow.add_edge("memory_synthesizer", "reporter")


def _build_persistence_components():
    """Use Postgres persistence when configured; otherwise fallback to in-memory saver."""
    db_uri = os.getenv("PE_MAS_LANGGRAPH_DB_URI", "").strip()
    if not db_uri:
        return MemorySaver(), None
    if PostgresSaver is None:
        print("⚠️ PE_MAS_LANGGRAPH_DB_URI set but langgraph PostgresSaver is unavailable. Fallback to MemorySaver.")
        return MemorySaver(), None

    checkpointer = PostgresSaver.from_conn_string(db_uri)
    try:
        checkpointer.setup()
    except Exception as e:
        print(f"⚠️ PostgresSaver setup failed ({e}). Fallback to MemorySaver.")
        return MemorySaver(), None

    store = None
    if PostgresStore is not None:
        try:
            store = PostgresStore.from_conn_string(db_uri)
            store.setup()
        except Exception as e:
            print(f"⚠️ PostgresStore setup failed ({e}). Continuing without long-term store injection.")
            store = None
    return checkpointer, store


checkpointer, store = _build_persistence_components()

# Compile with HITL configuration
# [AUTO MODE] Removed interrupts for fully autonomous execution
app = workflow.compile(
    checkpointer=checkpointer,
    store=store,
    interrupt_before=["selector", "simulator", "reporter", "correction"]
)

# Headless app for automatic curriculum/autonomous exploration loops.
app_headless = workflow.compile(
    checkpointer=checkpointer,
    store=store,
)
