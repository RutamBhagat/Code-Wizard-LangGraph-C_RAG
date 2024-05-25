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
            If the document contains any keyword(s) from the question or if the document semantically similar to the question, grade the document as relevant
            Answer as 'yes' or 'no' to indicate whether the document is relevant to the question"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system),
        HumanMessage(
            content="""Retrieved document: \n\n {document} \n\n User question: {question}"""
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

if __name__ == "__main__":
    question = "agent memory?"
    doc_text = "Long-term memory as the external vector store that the agent can attend to at query time, accessible via fast retrieval."
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    print(res)
