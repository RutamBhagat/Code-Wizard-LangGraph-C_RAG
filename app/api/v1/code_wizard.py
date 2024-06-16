import os
import json
import time
from fastapi import Body, APIRouter
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.graph import c_rag_app

router = APIRouter(prefix="/code_wizard", tags=["code_wizard"])


class Message(BaseModel):
    id: str
    role: str
    content: str


class RequestBody(BaseModel):
    chat_id: str
    chat_history: List[Message]

@router.post("/")
async def code_wizard(request_body: RequestBody = Body(...)):
    question = request_body.chat_history[-1].content
    # This is just for debugging to be removed in production
    with open(os.path.join(os.environ["PYTHONPATH"], "body.md"), "w") as f:
        json.dump({
            "chat_id": request_body.chat_id,
            "question": question
            }, f)

    start_time = time.time()
    config = {"configurable": {"thread_id": request_body.chat_id}}
    res = ""
    for event in c_rag_app.stream(input={"question": question}, config=config):
        for v in event.values():
            res = v
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")
    return res
