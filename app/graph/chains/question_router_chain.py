from typing import Literal
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vector store",
    )


llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
structured_llm_router = llm.with_structured_output(RouteQuery)

template = """You route user questions between a vectorstore and web search based on the conversation context. Use the vectorstore for LangChain-related queries covering:

                - Core Components: Model I/O, Prompts, Chat/LLM interfaces, Retrieval, Document loaders, Text splitters, Embeddings, Vectorstores, Retrievers, Tools, Agents, Chains, Memory, Callbacks
                - Off-the-shelf chains for high-level tasks
                - Libraries: langchain-core, langchain-community, langchain, langgraph, langserve
                - LangChain Expression Language (LCEL)

                Use web search for all other topics."""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}"),
    ]
)


question_router_chain = router_prompt | structured_llm_router
