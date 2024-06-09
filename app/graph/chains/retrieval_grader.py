from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeDocuments(BaseModel):
    """for relevance check on retrieved documents"""

    is_document_relevant: bool = Field(
        description="Documents are relevant to the question, 'True' or 'False'"
    )


llm = ChatOpenAI(temperature=0)

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

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # question = "LangChain Expression Language"
    question = "How to make pizza"
    chat_history = [HumanMessage(content="How to make pizza")]
    doc_text = """LangChain Expression Language, or LCEL, is a declarative way to chain LangChain components. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production). To highlight a few of the reasons you might want to use LCEL:"""
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "chat_history": chat_history, "document": doc_text}
    )
    print(res)
