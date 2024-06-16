from fastapi.middleware.cors import CORSMiddleware


def middleware():
    origins = ["*"]
    return CORSMiddleware(
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
