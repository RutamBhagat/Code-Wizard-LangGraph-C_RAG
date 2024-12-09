from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware
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


@app.head("/health")
@app.get("/health")
async def health_check():
    return {"status": "ok"}


app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
