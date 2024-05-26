from dotenv import load_dotenv, find_dotenv
from app.graph.graph import c_rag_app

_ = load_dotenv(find_dotenv())

if __name__ == "__main__":
    print("Hello C_RAG with LangGraph")
    res = c_rag_app.invoke(input={"question": "What is Agent Memory?"})
    print(res)
