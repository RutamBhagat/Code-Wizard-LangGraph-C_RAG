from langchain import hub
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model_name="mixtral-8x7b-32768")
prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()
