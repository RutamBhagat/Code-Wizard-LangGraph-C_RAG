from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv, find_dotenv
from starlette.middleware.cors import CORSMiddleware
from app.api import router as api_router

_ = load_dotenv(find_dotenv())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/health_check")
async def health_check():
    return {"status": "ok"}


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
