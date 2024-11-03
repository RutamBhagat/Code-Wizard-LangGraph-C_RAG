import time
from fastapi import Body, APIRouter
from pydantic import BaseModel
from app.graph.graph import c_rag_app

router = APIRouter(prefix="/code_wizard", tags=["code_wizard"])


class Message(BaseModel):
    id: str
    role: str
    content: str


class RequestBody(BaseModel):
    chat_id: str
    question: str


@router.post("/")
async def code_wizard(request_body: RequestBody = Body(...)):
    question = request_body.question
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
