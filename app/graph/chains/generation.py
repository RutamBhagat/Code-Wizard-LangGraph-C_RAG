from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context and the chat history to answer the question. 
If you don't know the answer based on the provided information, just say that you don't know. 
Use three sentences maximum and keep the answer concise unless the question asks for more details.

Question: {question}
Chat History: {chat_history}
Context: {context}
Answer:"""
prompt = PromptTemplate.from_template(prompt)
generation_chain = prompt | llm | StrOutputParser()
