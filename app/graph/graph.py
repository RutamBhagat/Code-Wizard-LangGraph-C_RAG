import sqlite3
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from app.graph.state import GraphState
from app.graph.consts import RETRIEVE, GENERATE, WEB_SEARCH, ENHANCED_QUERY
from app.graph.nodes import (
    generate,
    web_search,
    retrieve_documents_node,
    generate_enhanced_query_node,
)
from app.graph.chains.router import question_router, RouteQuery

_ = load_dotenv(find_dotenv())


def route_question(state: GraphState) -> str:
    source: RouteQuery = question_router.invoke(
        {"question": state.enhanced_query, "chat_history": state.chat_history}
    )
    if source.datasource == "web_search":
        return WEB_SEARCH
    else:
        return RETRIEVE


# Create the workflow without the SQLite connection
workflow = StateGraph(GraphState)

# Node Definition
workflow.add_node(ENHANCED_QUERY, generate_enhanced_query_node)
workflow.add_node(RETRIEVE, retrieve_documents_node)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

# Graph flow
workflow.set_entry_point(ENHANCED_QUERY)
workflow.add_conditional_edges(ENHANCED_QUERY, route_question, {RETRIEVE, WEB_SEARCH})
workflow.add_edge(RETRIEVE, WEB_SEARCH)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)


# Create a function to get a new graph instance with its own SQLite connection
def get_graph_instance():
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    memory.setup()  # Create tables if they don't exist
    return workflow.compile(checkpointer=memory)
