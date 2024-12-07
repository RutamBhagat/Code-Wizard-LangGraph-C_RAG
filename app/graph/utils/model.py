from langchain_openai import ChatOpenAI
from app.graph.consts import MODEL_NAME

model = ChatOpenAI(model=MODEL_NAME, temperature=0)
