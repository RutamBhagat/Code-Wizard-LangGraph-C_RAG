import sqlite3
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from app.graph.state import GraphInputState, GraphOutputState, GraphState
from app.graph.consts import (
    SUMMARIZE_CONVERSATION_NODE,
    ENHANCED_QUERY_NODE,
    RETRIEVE_NODE,
    WEB_SEARCH_NODE,
    GENERATE_NODE,
)
from app.graph.nodes import (
    summarize_conversation_node,
    generate_enhanced_query_node,
    web_search_node,
    retrieve_documents_node,
    generate_node,
)
from app.graph.chains.question_router_chain import question_router_chain, RouteQuery


_ = load_dotenv(find_dotenv())


def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """Save graph visualization to file."""
    with open(filename, "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())


def route_question(state: GraphState) -> str:
    source: RouteQuery = question_router_chain.invoke(
        {"question": state.enhanced_query, "chat_history": state.chat_history}
    )
    if source.datasource == "web_search":
        return WEB_SEARCH_NODE
    else:
        return RETRIEVE_NODE


# Create the builder without the SQLite connection
builder = StateGraph(GraphState, input=GraphInputState, output=GraphOutputState)

# Node Definition
builder.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation_node)
builder.add_node(ENHANCED_QUERY_NODE, generate_enhanced_query_node)
builder.add_node(RETRIEVE_NODE, retrieve_documents_node)
builder.add_node(WEB_SEARCH_NODE, web_search_node)
builder.add_node(GENERATE_NODE, generate_node)

# Graph flow
builder.add_edge(START, SUMMARIZE_CONVERSATION_NODE)
builder.add_edge(SUMMARIZE_CONVERSATION_NODE, ENHANCED_QUERY_NODE)
builder.add_conditional_edges(
    ENHANCED_QUERY_NODE,
    route_question,
    path_map={WEB_SEARCH_NODE: WEB_SEARCH_NODE, RETRIEVE_NODE: RETRIEVE_NODE},
)
builder.add_edge(RETRIEVE_NODE, GENERATE_NODE)
builder.add_edge(WEB_SEARCH_NODE, GENERATE_NODE)
builder.add_edge(GENERATE_NODE, END)

# Checkpointer
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn).setup()
graph = builder.compile(checkpointer=memory)
save_graph_visualization(graph)
