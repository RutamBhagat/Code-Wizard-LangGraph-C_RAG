from langchain import hub
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0, model="llama3-70b-8192")
prompt = hub.pull("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()
