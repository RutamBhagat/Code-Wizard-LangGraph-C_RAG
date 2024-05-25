from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeDocuments(BaseModel):
    """for relevance check on retrieved documents"""

    is_document_relevant: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question.
            If the document contains keyword(s) or if the document semantically matches the question grade it as relevant
            Give a is_document_relevant 'yes' or 'no' answer to indicate whether the document is relevant to the question"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system),
        HumanMessage(
            content="""Retrieved document: \n\n {document} \n\n User question: {question}"""
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
