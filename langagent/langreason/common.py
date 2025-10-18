from langagent.reasoner_base import Evaluator
import datasets
import re
import logging
import json
from langagent.base_llm import DETERMINISTIC_TEMPERATURE

logger = logging.getLogger(__name__)

def strip_num(num):
    """Aim to get a number parsable by Python's float()"""
    return num.strip("*").strip("`").strip("'").strip('"').strip()

def extract_numerical_answer(base_model, solution):

    base_model.sys_prompt = """Given a science or math problem and a corresponding solution, your task is to retrieve a numerical answer from the given solution if it exists. ONLY output the numerical answer as a plain number—parsable by Python’s float(), with no extra characters, commas, or symbols. If the solution does not contain a numerical answer, output an empty string."""
    user_message = "Solution:\n" + solution
    answer = base_model(user_message, role=None, temperature=DETERMINISTIC_TEMPERATURE, max_length=32768, max_new_tokens=20, enable_thinking=False).text.lower().strip()
    return answer

def retrieve_answer_from_gsm8k(example: dict) -> str:
    return re.match(r'[\S\s]*#### (.*)$', example['answer'])[1]

class QAEvaluator(Evaluator):
    def __init__(self, 
                 answer_extractor) -> None:
        pass

def load_qa_dataset(dataset_name):
    if dataset_name == "gsm8k":
        full_dataset = list(datasets.load_dataset('gsm8k', 'main', split='test'))
        for example in full_dataset:
            example["answer"] = retrieve_answer_from_gsm8k(example)
    elif dataset_name == "math500":
        # load from data/math500_float_answer_only_qa.jsonl
        with open("data/math500_float_answer.jsonl", "r") as f:
            full_dataset = [json.loads(line) for line in f]
    
    return full_dataset

def eval_output(answer, output):
    answer = answer.replace(",", "")
    assert output is not None 
    if output == "":
        return False

    try:
        output = int(output)
        answer = int(answer)
        
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer

def get_incorrect_df(full_dataset, results_from_file, incorrect_indices):
    # print(f"Run ID: {run_id}, Accuracy: {accuracy:.4f}, Correct Count: {correct_count}, Total Examples: {len(full_dataset)}")
    final_traces = [results_from_file.results[i][0] for i in incorrect_indices]

    df_rap_incorrect = pd.DataFrame({"index": incorrect_indices, \
        "question": [full_dataset[i]['question'] for i in incorrect_indices],\
        "answer": [full_dataset[i]['answer'] for i in incorrect_indices],
        "output": [retrieve_answer(final_trace[-1].state) for final_trace in final_traces],
        "final_trace": final_traces})
    return df_rap_incorrect

def get_correct_df(full_dataset, results_from_file, incorrect_indices):
    correct_indices = [i for i in range(len(results_from_file.results)) if i not in incorrect_indices]
    final_traces = [results_from_file.results[i][0] for i in correct_indices]
    df_rap_correct = pd.DataFrame({"index": correct_indices, \
        "question": [full_dataset[i]['question'] for i in correct_indices],\
        "answer": [full_dataset[i]['answer'] for i in correct_indices],
        "output": [retrieve_answer(final_trace[-1].state) for final_trace in final_traces],
        "final_trace": final_traces})
    return df_rap_correct