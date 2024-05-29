from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq


class GradeHallucinations(BaseModel):
    """Boolean value for hallucination present in the generation"""

    is_grounded: bool = Field(
        description="Answer is grounded in / supported by a set of documents, 'True' or 'False'"
    )


llm = ChatGroq(temperature=0, model="llama3-70b-8192", max_tokens=8192)

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of documents.
            Give a boolean value 'True' or 'False'. 'True" means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
