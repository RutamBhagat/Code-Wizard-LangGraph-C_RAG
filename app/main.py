import os
import asyncio
import time
from dotenv import load_dotenv, find_dotenv
from app.graph.graph import c_rag_app

load_dotenv(find_dotenv())


async def get_response(message):
    start_time = time.time()
    res = await c_rag_app.ainvoke(input={"question": message})
    end_time = time.time()
    time_taken = end_time - start_time
    return message, res, time_taken


async def main():
    messages = [
        "What is the purpose of the LangChain framework?",
        "What is Retrieval Augmented Generation (RAG)?",
        "What are the core features of chatbots?",
        "What are the two main ways to use tools in LangChain?",
        "What are the key features of LCEL that make it beneficial for building apps with LLMs?",
    ]
    print("Hello C_RAG with LangGraph")
    coroutines = [get_response(message) for message in messages]
    results = await asyncio.gather(*coroutines)

    with open(os.path.join(os.environ["PYTHONPATH"], "results.md"), "w") as f:
        for question, response, time_taken in results:
            f.write(f"Question: {question}\n")
            f.write(f"Time Taken: {time_taken:.2f} seconds\n")
            f.write(f"Response: {response['generation']}\n\n\n\n")


if __name__ == "__main__":
    asyncio.run(main())
