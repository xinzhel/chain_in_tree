from langchain.prompts import PromptTemplate
BASE_ACT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant Context: {context} 
Question: {question}{scratchpad}"""

BASE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. \
    You will be given a previous reasoning trial in which you were given access to relevant context and a question to answer. \
    You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or \
    there is a phrasing discrepancy with your provided answer and the answer key. \
    In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. \
    Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Relevant Context: {context}
Question: {question}{scratchpad}

Reflection:"""

base_act_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = BASE_ACT_INSTRUCTION,
                        )

base_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "context", "question", "scratchpad"],
                        template = BASE_REFLECT_INSTRUCTION,
                        )