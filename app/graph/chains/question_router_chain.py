from typing import Literal
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.graph.consts import MODEL_NAME
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        default="vectorstore", description="The data source to route the query to"
    )

    class Config:
        use_enum_values = True


structured_llm_router = llm.with_structured_output(RouteQuery)

template = """You are a specialized routing assistant that directs questions to the most appropriate data source.

AVAILABLE DATA SOURCES:
1. Vectorstore (Primary Source)
2. Web Search (Secondary Source)

VECTORSTORE DOMAIN EXPERTISE:
1. LangChain Core Components:
   - Model I/O and Interfaces (Chat/LLM)
   - Prompts and Templates
   - Document Processing (Loaders, Splitters)
   - Retrieval Systems (Embeddings, Vectorstores, Retrievers)
   - Agent Components (Tools, Agents, Chains)
   - System Features (Memory, Callbacks)

2. LangChain Ecosystem:
   - Framework Libraries (langchain-core, langchain-community, langchain)
   - Extensions (langgraph, langserve)
   - LangChain Expression Language (LCEL)
   - Off-the-shelf chains and applications

ROUTING GUIDELINES:
- Default to vectorstore for LangChain-related queries
- Use web_search for:
  - Non-LangChain topics
  - Current events or recent updates
  - General programming questions
  - Topics not covered in LangChain documentation

RESPONSE FORMAT:
You must respond with exactly one of these options:
- "vectorstore"
- "web_search"

Consider both the chat history and current question context for optimal routing.
"""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Question: {question}"),
    ]
)


question_router_chain = router_prompt | structured_llm_router
