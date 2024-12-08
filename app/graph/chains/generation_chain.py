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

# The following is now part of the first human message.
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
        ("human", template),  # System message converted to human message
        MessagesPlaceholder(variable_name="chat_history"),  # Keep as is
        (
            "human",
            "{question}",
        ),  # Keep as is - this is just for separation within the human prompt
    ]
)
generation_chain = ANSWER_PROMPT | llm | StrOutputParser()
