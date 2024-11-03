from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeHallucinations(BaseModel):
    """Boolean value for hallucination present in the generation"""

    is_grounded: bool = Field(
        description="Answer is grounded in / supported by a set of documents, 'True' or 'False'"
    )


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

template = """Assess whether an LLM generation is grounded in / supported by a set of documents or facts.
To consider the generation grounded, it should be consistent with and supported by the information provided in the set of documents or facts, 
without introducing any new unsupported information or contradicting the given facts.

Reply with 'True' or 'False' to indicate is_generation_grounded."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
