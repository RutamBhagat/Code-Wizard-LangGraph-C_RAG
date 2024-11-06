import sqlite3
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from app.graph.state import GraphState
from app.graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH, ADD_ANSWER
from app.graph.nodes import retrieve, generate, web_search
from app.graph.chains.router import question_router, RouteQuery

_ = load_dotenv(find_dotenv())


def route_question(state: GraphState) -> str:
    question = state.question
    chat_history = state.chat_history or []
    source: RouteQuery = question_router.invoke(
        {"question": question, "chat_history": chat_history}
    )
    if source.datasource == "web_search":
        return WEB_SEARCH
    else:
        return RETRIEVE


# Create the workflow without the SQLite connection
workflow = StateGraph(GraphState)

# Node Definition
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

# Graph flow
workflow.set_conditional_entry_point(
    route_question, path_map={WEB_SEARCH: WEB_SEARCH, RETRIEVE: RETRIEVE}
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(ADD_ANSWER, END)


# Create a function to get a new graph instance with its own SQLite connection
def get_graph_instance():
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    memory.setup()  # Create tables if they don't exist
    return workflow.compile(checkpointer=memory)
