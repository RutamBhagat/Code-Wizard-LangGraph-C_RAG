from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from app.graph.consts import MODEL_NAME
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context and the chat history to answer the question. 
If you don't know the answer based on the provided information, just say that you don't know. 
Try to keep the answer concise unless the question asks for more details or code is requested 
(only provide code if the context contains the code or else ignore providing code).
Please provide your responses in valid markdown format.
Chat History: {messages}
Context: {context}
Question: {question}
Answer:"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", template),
        MessagesPlaceholder(variable_name="messages"),
        (
            "human",
            "{question}",
        ),
    ]
)
generation_chain = ANSWER_PROMPT | llm | StrOutputParser()
