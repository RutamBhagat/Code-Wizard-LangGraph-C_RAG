from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):
    """Boolean value for hallucination present in the generation"""

    is_answer_valid: bool = Field(
        description="Answer addresses the question, 'True' or 'False'"
    )


llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question.
            Give a boolean value 'True' or 'False'. 'True" means that the answer resolves the question"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Users Question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
