import sys
sys.path.append("../..")
import os
import time
import shutil
from tqdm import tqdm
import diskcache as dc
import torch
from langagent.langreason.common import eval_output, load_qa_dataset
from langagent.base_llm import HfModel, InferenceLogger, HfChatModel, DETERMINISTIC_TEMPERATURE
from langagent.eval import ResultDictToJsonl
from langagent.log import setup_logging
# Add you own Hf token
# from huggingface_hub import login
# hf_token = ""
# login(token = '')

def extract_numerical_answer(base_model, question, solution):

    base_model.sys_prompt = """You are given a science or math problem along with its solution.
Your task is to **extract the numerical answer from the solution if it is explicitly provided**.

#### Rules:

1. If the solution does **not** contain a numerical answer, output an empty string `""`.
2. You must **not** generate your own answer or infer one that is not directly stated in the solution.
3. If the solution contains a numerical answer, output **only that number**, in plain format parsable by Python’s `float()`.

   * Do **not** include units, symbols, commas, words, or extra characters.
   * Do **not** add explanations or formatting.

#### Output format:

* A single number (e.g., `42` or `3.14`)
* Or an empty string `""`

---

#### ✅ Valid outputs:

* `42`
* `3.14`
* `0`
* `""` (if no numerical answer is present)

#### ❌ Invalid outputs:

* `"42 apples"`
* `"The answer is 42"`
* `"3,140"` (commas not allowed)
* `"3.14 meters"`
* `"No answer provided"`"""
    user_message = "Question: " + question + "\nSolution:\n" + solution
    answer = base_model(user_message, role=None, temperature=DETERMINISTIC_TEMPERATURE, max_length=32768, max_new_tokens=20, enable_thinking=False).text.lower().strip()
    return answer
def check_device(dataset_name):
    device_name = torch.cuda.get_device_name(0).lower()
    if dataset_name == "math500":
        assert "a100" in device_name
    elif dataset_name == "gsm8k":
        assert "l40s" in device_name
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
def get_accuracy(full_dataset, results_file, verbose=False):
    correct_count = 0
    incorrect_indices = []
    for example_idx in range(len(full_dataset)):
        example = full_dataset[example_idx]
        # ground truth
        answer = example['answer']
        if verbose:
            print("answer: ", answer)
        
        # model output
        output = results_file.results[example_idx]['label']
        if verbose:
            print("output: ", output)
        
        # caculate accuracy
        correct = eval_output(answer, output)
        if not correct:
            incorrect_indices.append(example_idx)
        correct_count += correct
        example_idx += 1

    accuracy = correct_count / len(full_dataset)
    return {"accuracy": accuracy, "correct_count": correct_count, "total_examples": len(full_dataset), "incorrect_indices": incorrect_indices}

def main(dataset_name, model_name, extract_model_name=None):
    
    # dataset_name = "gsm8k"#"math500"
    if dataset_name == "math500":
        size_eval = 100 # default: None
    else:
        size_eval = 100
    method = "rest_cot" #"cot"
    run_id = dataset_name + "_"+ method
    override_log_file = True
    override_result_file = True 
    override_cache = True
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # "Qwen/Qwen3-32B-AWQ" #
    device_name = torch.cuda.get_device_name(0).lower()
    max_length=32768

    #check_device(dataset_name)

    # create root dir
    root_dir = f"{model_name.split('/')[-1]}_results/{run_id}"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    logger = setup_logging(run_id, root_dir, add_console_handler=True, verbose=True)
    max_length=32768
    verbose = True

    if method == "rest_cot":
        sys_prompt = None
    else:
        sys_prompt = """Solve math problems step by step. Enclose all reasoning inside <think>...</think>. After </think>, output only the final answer as a plain number—parsable by Python’s float(), with no extra characters, commas, or symbols."""

    # reasoner = CoTReasoner(base_model, temperature=temperature, n_sc=n_sc, bs=batch_size)
    bs = 1
    temperature = 1e-6
    if temperature == DETERMINISTIC_TEMPERATURE:
        print("Using deterministic temperature for inference.")
    
    if model_name == "meta-llama/Meta-Llama-3-8B":
        base_model = HfModel.load_from_hf(model_name, device="cuda")
    elif model_name == "Qwen/Qwen3-32B-AWQ" or model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        base_model = HfChatModel.load_from_hf(model_name, device="cuda", sys_prompt=sys_prompt)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if extract_model_name is not None:
        extract_model = HfChatModel.load_from_hf(extract_model_name, device="cuda")
    # base_model = HfChatModel.load_from_hf("Qwen/Qwen3-0.6B", device="mps", sys_prompt=sys_prompt)

    inference_logger = InferenceLogger(run_id='', root_dir=root_dir, override=override_log_file)
    base_model.inference_logger = inference_logger
    results_file = ResultDictToJsonl(run_id='', root_dir=root_dir, override=override_result_file)
    example_idx = 0

    full_dataset = load_qa_dataset(dataset_name)
    if size_eval is not None:
        full_dataset = full_dataset[:size_eval]
    time_begin = time.time()
    for example in tqdm(full_dataset):
        example_txt = example["question"]
        # ~~~~~~~ reasoner (BEGIN)~~~~~~~
        if method == "rest_cot":
            usr_prompt = """Given a science problem, your task is to answer the question step-by-step in a clear and specific manner.
The format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"
Please complete the answer step-by-step, and finally outline the final answer.
Problem: """ + example_txt + "\nSolution:"
            model_output = base_model(usr_prompt,  temperature=temperature, max_length=max_length, max_new_tokens=1024).text.strip()
            logger.debug(f"Raw Output: {model_output}")
            extracted_output = extract_numerical_answer(base_model if extract_model_name is None else extract_model, example_txt, model_output)
            if extracted_output not in model_output:
                logger.warning(f"Extracted answer {extracted_output} not in model output {model_output}")
                extracted_output = ""
            logger.debug(f"Extracted Answer: {extracted_output}")
            model_output = "<think></think>\n " + extracted_output
        else:
            usr_prompt = example_txt
            model_output = base_model(usr_prompt,  temperature=temperature, max_length=max_length).text.strip()
            model_output = model_output.strip() if model_output.endswith(".") else model_output + "."
        # ~~~~~~~ reasoner (END)~~~~~~~

        # ~~~~~~~ save results (BEGIN) ~~~~~~~
        results_file.append_result(model_output, truth=example["answer"]) # all the llm output
        # ~~~~~~~ save results (END) ~~~~~~~
        example_idx += 1
    end_time = time.time()


    # Logging
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Model: {model_name}")
    logger.info(f"The max length is: {max_length}")
    device_name = torch.cuda.get_device_name(0).lower()
    logger.info(f"Device name: {device_name}")
    logger.info(f"Total time: {end_time - time_begin}")
    metric = get_accuracy(full_dataset, results_file)
    logger.info(f"Run ID: {run_id}, Accuracy: {metric['accuracy']:.4f}, Correct Count: {metric['correct_count']}, Total Examples: {metric['total_examples']}")
    token_usage = str(base_model.inference_logger.get_metrics_by_prefix("default"))
    logger.info(f"default: \t {token_usage}")


main(dataset_name="math500", model_name="meta-llama/Meta-Llama-3-8B", extract_model_name="Qwen/Qwen3-32B-AWQ") # "meta-llama/Meta-Llama-3-8B-Instruct"
main(dataset_name="gsm8k", model_name="meta-llama/Meta-Llama-3-8B", extract_model_name="Qwen/Qwen3-32B-AWQ")
# main(dataset_name="math500", model_name="Qwen/Qwen3-32B-AWQ")
