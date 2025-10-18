from typing import List, Tuple, Dict
from langagent.framework_config import qa_rest
import pandas as pd
class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]=None):
        self.input_variables = set(input_variables) if input_variables else None
        self.template = template

    def format(self, **kwargs) -> str:
        if self.input_variables is not None: 
            missing = self.input_variables - kwargs.keys()
            if missing:
                raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

# the prompt is adapted from REST-MCTS https://arxiv.org/pdf/2406.03816
rest_qa = {
    "actor": PromptTemplate(qa_rest["policy_sys"]+"""

Problem: {problem}
Existing Steps:
{existing_steps}
Output:"""),

    "solution_verifier": PromptTemplate(
"""Given a science or math problem, a corresponding step-by-step solution, and the true answer of the problem, \
    your task is to verify the answer obtained in the solution with the real answer. If the answer obtained in \
    the solution is equivalent to the real one, output '1', otherwise output '0'.  \
    \nProblem: {problem} \
    \nSolution: {solution} \
    \nReal Answer: {real_answer}"""
)
}

if __name__ == "__main__":
    print(rest_qa["solution_verifier"].format(
        problem="What is the capital of France?",
        solution="The capital of France is Paris.",
        real_answer="Paris"
    ))


    