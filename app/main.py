from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="RAG Province")

app.include_router(router)