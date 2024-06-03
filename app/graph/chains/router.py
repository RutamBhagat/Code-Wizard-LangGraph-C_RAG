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

system_prompt = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore is related to LangChain and it provides information about the following topics:
<<<
    Components: These are composable tools and integrations for working with language models. They are modular and easy-to-use, whether you are using the rest of the LangChain framework or not. Some of the main components include:
        Model I/O: Formatting and managing language model input and output.
        Prompts: Formatting for LLM inputs that guide generation.
        Chat models: Interfaces for language models that use chat messages as inputs and returns chat messages as outputs.
        LLMs: Interfaces for language models that use plain text as input and output.
        Retrieval: Interface with application-specific data for e.g. RAG.
        Document loaders: Load data from a source as Documents for later processing.
        Text splitters: Transform source documents to better suit your application.
        Embedding models: Create vector representations of a piece of text, allowing for natural language search.
        Vectorstores: Interfaces for specialized databases that can search over unstructured data with natural language.
        Retrievers: More generic interfaces that return documents given an unstructured query.
        Composition: Higher-level components that combine other arbitrary systems and/or or LangChain primitives together.
        Tools: Interfaces that allow an LLM to interact with external systems.
        Agents: Constructs that choose which tools to use given high-level directives.
        Chains: Building block-style compositions of other runnables.
        Memory: Persist application state between runs of a chain.
        Callbacks: Log and stream intermediate steps of any chain. (source (https://python.langchain.com/v0.1/docs/modules/))

    Off-the-shelf chains: These are built-in assemblages of components for accomplishing higher-level tasks. They make it easy to get started and customize existing chains and build new ones. (source (https://python.langchain.com/v0.1/docs/integrations/document_loaders/tomarkdown/))


    LangChain Libraries: The LangChain libraries themselves are made up of several different packages:
        langchain-core: Base abstractions and LangChain Expression Language.
        langchain-community: Third party integrations.
        langchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
        langgraph: An extension of langchain aimed at building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.
        langserve: A package to deploy LangChain chains as REST APIs. (source (https://python.langchain.com/v0.2/docs/concepts/))


    LangChain Expression Language (LCEL): LCEL is a declarative way to compose chains. It was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains. (source (https://python.langchain.com/v0.1/docs/integrations/document_loaders/tomarkdown/))


Please note that these are just the main components and packages. LangChain also provides many other features and tools to help you build powerful applications. For more detailed information, you might want to check the LangChain documentation.
>>>
Use the vectorstore for questions on these topics. For other questions, use the web search.
"""


router_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)

question_router = router_prompt | structured_llm_router
