from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """for relevance check on retrieved documents"""

    is_document_relevant: bool = Field(
        description="Documents are relevant to the question, 'True' or 'False'"
    )


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

template = """Determine the relevance of a retrieved document to a user's question while taking into account the recent chat history context.
To consider a document relevant, it should provide explicit instructions, steps, or procedural information directly related to the key terms or concepts mentioned in the question or the recent chat history.
Mentioning the key terms or concepts alone is not enough; the context should be about the process or task described in the question.
Additionally, consider synonyms, paraphrases, or alternative phrasings of the key terms or concepts, as well as descriptions or explanations that provide the necessary information to answer the question, even if they don't use the exact terms. 
Else determine that the document is not relevant to the question.

Reply with 'True' or 'False' to indicate is_document_relevant."""


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "Retrieved document: \n\n {document} \n\n User question: {question}",
        ),
    ]
)


retrieval_grader = grade_prompt | structured_llm_grader
