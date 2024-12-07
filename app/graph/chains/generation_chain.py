from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context and the chat history to answer the question. 
If you don't know the answer based on the provided information, just say that you don't know. 
Try to keep the answer concise unless the question asks for more details or code is requested.
Please provide your responses in valid markdown format.
Chat History: {chat_history}
Context: {context}
Question: {question}
Answer:"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)
generation_chain = ANSWER_PROMPT | llm | StrOutputParser()
