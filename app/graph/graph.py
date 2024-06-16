from typing import Any, Dict, Sequence
from dotenv import load_dotenv, find_dotenv
from langchain.schema import HumanMessage
from langchain_core import chat_history
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver


from app.graph.state import GraphState
from app.graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH, ADD_ANSWER
from app.graph.nodes import retrieve, grade_documents, generate, web_search
from app.graph.chains.hallucination_grader import hallucination_grader
from app.graph.chains.answer_grader import answer_grader
from app.graph.chains.router import question_router, RouteQuery

_ = load_dotenv(find_dotenv())


def decide_to_generate(state) -> str:
    print("Assessing if we need a web search or we should generate the answer.")
    if state.is_web_search_needed:
        print(
            "----DECISION: NOT ALL DOCUMENT ARE RELEVANT TO THE QUESTION, PERFORMING WEB SEARCH"
        )
        return WEB_SEARCH
    else:
        print("----DECISION: ALL DOCUMENTS ARE RELEVANT TO THE QUESTION")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("Assessing if the generation is grounded in the documents and question.")
    question = state.question
    documents = state.documents
    generation = state.generation
    chat_history = state.chat_history

    is_grounded = hallucination_grader.invoke(
        {"documents": documents, "generation": generation, "chat_history": chat_history}
    ).is_grounded

    if is_grounded:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        is_answer_valid = answer_grader.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "generation": generation,
            }
        ).is_answer_valid

        if is_answer_valid:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
def add_answer(state: GraphState) -> Dict[str, Any]:
    print("____ADD ANSWER____")
    generation = state.generation
    chat_history = state.chat_history or []
    chat_history.append(HumanMessage(content=generation))
    return { "chat_history": chat_history, "generation": generation}


def route_question(state: GraphState) -> str:
    print("____ROUTE QUESTION____")
    question = state.question
    chat_history = state.chat_history or []
    source: RouteQuery = question_router.invoke(
        {"question": question, "chat_history": chat_history}
    )
    if source.datasource == "web_search":
        return WEB_SEARCH
    else:
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(ADD_ANSWER, add_answer)

workflow.set_conditional_entry_point(route_question, path_map={WEB_SEARCH: WEB_SEARCH, RETRIEVE: RETRIEVE}) 
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={"useful": ADD_ANSWER, "not useful": WEB_SEARCH, "not supported": GENERATE},
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(ADD_ANSWER, END)

memory = SqliteSaver.from_conn_string(":memory:")
c_rag_app = workflow.compile(checkpointer=memory)
c_rag_app.get_graph().draw_mermaid_png(output_file_path="graph.png")
