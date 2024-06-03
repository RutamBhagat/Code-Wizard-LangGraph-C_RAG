from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vector store",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system_prompt = """Option 1 (vectorstore): A vectorstore with a large collection of documents about LangChain. LangChain is a framework for building applications using large language models (LLMs). The vectorstore contains:
- Overview and intro to LangChain
- Tutorials and guides for building LLM apps with LangChain
- Explanations of key LangChain components and concepts  
- API documentation for LangChain's Python libraries
- Info on LangChain's ecosystem like LangSmith, LangGraph, LangServe
- Security best practices for using LangChain
- Details on integrating third-party tools with LangChain
- Guidelines for contributing to LangChain

Option 2 (web_search): Access to search the vast information on the internet.

Your job is to carefully read each user's question. Then decide if the vectorstore or web search is the best place to find a relevant and accurate answer. Use the vectorstore for questions related to LangChain, since it has reliable LangChain documentation. 

But if the question is not about LangChain or not covered in the vectorstore, use web searches to find trustworthy sources to answer it effectively.

Use the vectorstore first for LangChain questions. Only use web searches when needed for questions the vectorstore can't fully answer.
"""


router_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)

question_router = router_prompt | structured_llm_router
