import time
from fastapi import Body, APIRouter
from pydantic import BaseModel
from app.graph.graph import get_graph_instance

router = APIRouter(prefix="/code_wizard", tags=["code_wizard"])


class Message(BaseModel):
    id: str
    role: str
    content: str


class RequestBody(BaseModel):
    chat_id: str
    question: str


@router.post("/")
def code_wizard(request_body: RequestBody = Body(...)):
    question = request_body.question
    start_time = time.time()
    config = {"configurable": {"thread_id": request_body.chat_id}}

    # Get a new graph instance for this request
    graph_instance = get_graph_instance()

    # Use the instance for this request
    res = graph_instance.invoke({"question": question}, config=config)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")

    return res
