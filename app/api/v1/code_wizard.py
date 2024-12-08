import time
from fastapi import Body, APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.graph.graph import graph

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

    # Use the instance for this request
    res = await graph.ainvoke({"question": question}, config=config)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")

    return res


@router.post("/stream")
async def code_wizard_stream(request: Request, request_body: RequestBody = Body(...)):
    question = request_body.question
    config = {"configurable": {"thread_id": request_body.chat_id}}

    async def event_generator():
        try:
            async for event in graph.astream_events(
                {"question": question}, config=config, version="v2"
            ):
                # Check if client is disconnected
                if await request.is_disconnected():
                    break
                # Filter for events related to the 'generate' node
                if (
                    event["event"] == "on_chat_model_stream"
                    and event["metadata"].get("langgraph_node", "") == "generate"
                ):
                    data = event["data"]
                    # Send the content of the AIMessageChunk
                    yield {
                        "event": "message",
                        "data": data["chunk"].content,
                    }
        except Exception as e:
            print(f"Error: {e}")
            yield {
                "event": "error",
                "data": "An error occurred during streaming.",
            }
        finally:
            # Send a completion message
            yield {"event": "end", "data": ""}

    return EventSourceResponse(event_generator())
