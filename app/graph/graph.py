from dotenv import load_dotenv, find_dotenv

from app.graph.nodes.generate import generate
from app.graph.nodes.web_search import web_search
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.grade_documents import grade_documents
from app.graph.state import GraphState
from langgraph.graph import END, StateGraph

_ = load_dotenv(find_dotenv())

RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
WEB_SEARCH = "web_search"
GENERATE = "generate"


def should_continue(state: GraphState) -> str:
    grade = grade_documents(state)
    if grade["is_web_search_needed"]:
        return WEB_SEARCH
    else:
        return GENERATE


flow = StateGraph(GraphState)
flow.add_node(RETRIEVE, retrieve)
flow.add_node(WEB_SEARCH, web_search)
flow.add_node(GENERATE, generate)
flow.set_entry_point(RETRIEVE)
flow.add_conditional_edges(RETRIEVE, should_continue)
flow.add_edge(WEB_SEARCH, GENERATE)
flow.add_edge(GENERATE, END)

c_rag_app = flow.compile()
c_rag_app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello C_RAG with LangGraph")
    res = c_rag_app.invoke(input={"input": "Agent Memory"})
    print(res)
