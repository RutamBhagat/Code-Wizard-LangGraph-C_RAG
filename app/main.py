from dotenv import load_dotenv, find_dotenv
from app.graph.graph import c_rag_app
import time
import os

load_dotenv(find_dotenv())

if __name__ == "__main__":
    messages = [
        "What is the purpose of the LangChain framework?",
        "What is Retrieval Augmented Generation (RAG)?",
        "What are the three broad approaches for information extraction using Large Language Models (LLMs)?",
        "What are the core features of chatbots?",
        "What are the two main ways to use tools in LangChain?",
        "What are the key features of LCEL that make it beneficial for building apps with LLMs?",
    ]
    print("Hello C_RAG with LangGraph")
    times = []
    for message in messages:
        start_time = time.time()
        res = c_rag_app.invoke(input={"question": message})
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Time for {message}: {end_time - start_time} seconds")

    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    print(f"Minimum time: {min_time} seconds")
    print(f"Maximum time: {max_time} seconds")
    print(f"Average time: {avg_time} seconds")

    with open(os.path.join(os.environ["PYTHONPATH"], "results.md"), "w") as f:
        f.write("# Results\n\n")
        f.write("## Minimum Time\n")
        f.write(f"Minimum time: {min_time} seconds\n")
        f.write("\n")
        f.write("## Maximum Time\n")
        f.write(f"Maximum time: {max_time} seconds\n")
        f.write("\n")
        f.write("## Average Time\n")
        f.write(f"Average time: {avg_time} seconds\n")
        f.write("\n\n\n")
        f.write("## Time for each message\n\n")
        for i, message in enumerate(messages):
            f.write(f"Time for ### {message}: \n{times[i]} seconds\n\n")
