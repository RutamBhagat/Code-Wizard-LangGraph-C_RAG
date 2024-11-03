from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    """Boolean value for hallucination present in the generation"""

    is_answer_valid: bool = Field(
        description="Answer addresses the question, 'True' or 'False'"
    )


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

structured_llm_grader = llm.with_structured_output(GradeAnswer)

template = """You are a grader assessing whether an answer addresses / resolves a question while taking into account the chat history context.
            Give a boolean value 'True' or 'False'. 'True" means that the answer resolves the question"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "user",
            "Users Question: \n\n {question} \n\n LLM generation: {generation}",
        ),
    ]
)


answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
