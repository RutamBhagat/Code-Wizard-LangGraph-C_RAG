from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class GradeDocuments(BaseModel):
    """for relevance check on retrieved documents"""

    is_document_relevant: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    explaination: str = Field(
        description="Explaination of why the document is relevent or not"
    )


llm = ChatOpenAI(temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """Your task is to determine the relevance of a retrieved document to a user's question.
            If the document mentions key terms, concepts, or contextual information related to the question, consider the document relevant, even if it does not explicitly answer the question.
            Reply with 'yes' or 'no' to indicate the document's relevance. Along with the explaination of why the document is relevent or not"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

if __name__ == "__main__":
    question = "Agent Memory"
    doc_text = """They also discussed the risks, especially with illicit drugs and bioweapons. They developed a test set containing a list of known chemical weapon agents and asked the agent to synthesize them. 4 out of 11 requests (36%) were accepted to obtain a synthesis solution and the agent attempted to consult documentation to execute the procedure. 7 out of 11 were rejected and among these 7 rejected cases, 5 happened after a Web search while 2 were rejected based on prompt only.
                    Generative Agents Simulation#
                    Generative Agents (Park, et al. 2023) is super fun experiment where 25 virtual characters, each controlled by a LLM-powered agent, are living and interacting in a sandbox environment, inspired by The Sims. Generative agents create believable simulacra of human behavior for interactive applications.
                    The design of generative agents combines LLM with memory, planning and reflection mechanisms to enable agents to behave conditioned on past experience, as well as to interact with other agents.

                    Memory stream: is a long-term memory module (external database) that records a comprehensive list of agents’ experience in natural language.
                """
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )
    print(res)
    # The res is always "no" why is that?
