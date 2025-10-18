import sys
sys.path.append('../..')
import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dataclasses import asdict
import shutil
print (f"Current working directory: {os.getcwd()}")
import json
import logging
import torch
import numpy as np
import diskcache as dc
from tqdm import trange, tqdm
from langagent.base_llm import InferenceLogger, VALID_ROLES_PREFIX, HfChatModel, HfModel, DETERMINISTIC_TEMPERATURE
from langagent.eval import TreeToJsonl, ResultDictToJsonl
from langagent.search.mcts import mcts, MCTSSearchConfig, get_result_from_mcts
from langagent.search.bfs import bfs_topk, BFSConfig
from langagent.framework_config import gsm8k_rap, qa_rest
from langagent.log import setup_logging
from langagent.sys_utils import is_running_in_jupyter
from langagent.langreason.mcts_utils import make_retrieve_answer, RAPWorldModel, RAPPolicy, RAPEvaluator, get_task_instruction, get_task_instruction_with_examples, BNEvaluator, get_accuracy, RestWorldModel, RestPolicy, RestEvaluator, RestEvaluator2, retrieve_answer_from_last_step
from langagent.langreason.common import QAEvaluator, eval_output, load_qa_dataset, retrieve_answer_from_gsm8k
# Add you own Hf token
# from huggingface_hub import login
# hf_token = ""
# login(token = '')
package_version = "v0.1.6"
def check_device(dataset_name):
    device_name = torch.cuda.get_device_name(0).lower()
    if dataset_name == "math500":
        assert "a100" in device_name
    elif dataset_name == "gsm8k":
        assert "l40s" in device_name
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def is_mcts_method(reasoning_method):
    return reasoning_method == "rest" or reasoning_method == "rap"

def main(dataset_name, model_name, eval_model_name, reasoning_method, add_continuation, bn_method, bn_model_name=None, eval_idx=[]):
    """
    bn_model_name: if add_continuation and bn_method is not None, bn_model_name is not provided, base_model is used
    """
    # dataset_name = "math500"# "gsm8k" #
    # eval_idx = [] #[] # default: []
    if dataset_name == "math500":
        head_size_eval = 100 # default: None
    else:
        head_size_eval = 100
    assert eval_idx == [] or max(eval_idx) < head_size_eval, f"There exists an example index in eval_idx that is larger than head_size_eval"

    # llm params
    max_length=32768
    device = "cuda" #"mps" # 
    # check_device(dataset_name)

    # search params - (General)
    # reasoning_method =  "rest"  # "rap"  # "bfs"
    n_actions = 3 # default 3 for rest, 2 for rap

    # search params - termination - (General)
    terminate_constraints= ['binary_sampling',] #'verify'] # 'reward_threshold'
    terminate_ORM_name = None # "RLHFlow/Llama3.1-8B-ORM-Deepseek-Data"
    if reasoning_method == "rap":
        terminal_gen_model_name = None
    else:
        terminal_gen_model_name = "Qwen/Qwen3-32B-AWQ" #None #
    r_terminating = None # 0.9 # can be used for search termination or/and determining terminal state
    sample_size_terminate = 10
    sample_threshold_terminate = 0.8
    sample_threshold_verify = 0.8
    depth_limit = 10
    if reasoning_method == "rap":
        force_terminating_on_depth_limit = True
    else:
        force_terminating_on_depth_limit = False
    terminate_on_terminal_node = True
    runtime_limit_before_iter = 3600

    if eval_model_name.startswith("meta-llama/Meta-Llama-3-8B"):
        think_for_usefulness = True
        think_for_correctness = True
        n_for_correctness = 5
        n_for_usefulness = 5
    elif eval_model_name == "Qwen/Qwen3-32B-AWQ":
        think_for_usefulness = False
        think_for_correctness = False
        n_for_correctness = 2
        n_for_usefulness = 1
    elif eval_model_name ==  "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        think_for_usefulness = None
        think_for_correctness = None
        n_for_correctness = None
        n_for_usefulness = None
    else:
        raise Exception

    # search params - (MCTS-specific)
    if reasoning_method == "rest":
        roll_out_steps = 2 # 10000 for rap, 2 for rest
        n_iters = 50 # default 10 for rap, 50 for rest
        n_action_for_simulate = n_actions 
        n_confidence = None
    elif reasoning_method == "rap":
        roll_out_steps = 10000
        n_iters = 10
        n_action_for_simulate = n_actions # 
        n_confidence = 3
    else:
        pass
    # search params (NOT FOUND USEFUL)
    check_action_sim = False  # ONLY for rest
    use_critic = False # ONLY for rest

    # continuation
    # add_continuation = True
    if add_continuation:
        # bn_method =   "direct" # None #"entropy" # "sc"
        reward_alpha = None #0.8
        reward_beta = None # 0.8
        
        if bn_method == "entropy":
            reward_gamma = 0.13 
            max_new_tokens_for_bn_eval = 1000
        elif bn_method == "sc":
            reward_gamma = 0.49
            max_new_tokens_for_bn_eval = None
        else: #"direct"
            reward_gamma = 0.7 
            max_new_tokens_for_bn_eval = None
        reward_gamma1 = None
        n_actions_for_bne = 3 if bn_method == "entropy" or bn_method == "sc" else None
        only_continuation_at_head = False
        
        max_try_for_bn_eval = 3
    else:
        bn_method = None
        bn_model_name = None
        reward_alpha = None
        reward_beta = None
        reward_gamma = None
        reward_gamma1 = None
        n_actions_for_bne = None
        only_continuation_at_head = None
        max_new_tokens_for_bn_eval = None
        max_try_for_bn_eval = None

    # search config
    common_config ={
        "package_version": package_version,
        "reasoning_method": reasoning_method,
        "n_actions": n_actions,
        "runtime_limit_before_iter": runtime_limit_before_iter,

        # model name
        "model_name": model_name,
        "eval_model_name": eval_model_name,
        "gpu_device": torch.cuda.get_device_name(0).lower(),

        # terminate
        "terminate_constraints": terminate_constraints,
        "terminate_ORM_name": terminate_ORM_name,
        "terminal_gen_model_name": terminal_gen_model_name,
        "r_terminating": r_terminating,
        "sample_size_terminate": sample_size_terminate,
        "sample_threshold_terminate": sample_threshold_terminate,
        "sample_threshold_verify": sample_threshold_verify,
        "depth_limit": depth_limit,
        "force_terminating_on_depth_limit": force_terminating_on_depth_limit,
        "terminate_on_terminal_node": terminate_on_terminal_node,

        # eval
        "think_for_usefulness": think_for_usefulness,
        "think_for_correctness": think_for_correctness,
        "n_for_correctness": n_for_correctness,
        "n_for_usefulness": n_for_usefulness,

        # continuation
        "bn_model_name": bn_model_name,
        "reward_alpha": reward_alpha,
        "reward_beta": reward_beta,
        "reward_gamma": reward_gamma,
        "reward_gamma1": reward_gamma1,
        "add_continuation": add_continuation,
        "bn_method": bn_method,
        "n_actions_for_bne": n_actions_for_bne,
        "only_continuation_at_head": only_continuation_at_head,
        "max_new_tokens_for_bn_eval": max_new_tokens_for_bn_eval,
        "max_try_for_bn_eval": max_try_for_bn_eval,
    }

    if is_mcts_method(reasoning_method):
        search_config = MCTSSearchConfig(
            w_exp = 1.0,
            cum_reward = np.mean,
            calc_q = max,
            n_iters = n_iters,
            roll_out_steps = roll_out_steps,

            output_trace_in_each_iter = True,
            use_critic=use_critic,
            n_action_for_simulate=n_action_for_simulate,
            n_confidence = n_confidence,
            **common_config
        )
    elif reasoning_method == "bfs":
        search_config = BFSConfig(**common_config)
    else:
        raise ValueError(f"Unknown inference method: {reasoning_method}") 

    # override 
    override_log_result = True
    if is_running_in_jupyter():
        HfModel.set_log_model_input(True)
        HfModel.set_log_model_output(True)
        HfChatModel.set_log_model_input(True)
        HfChatModel.set_log_model_output(True)
    else:
        HfModel.set_log_model_input(False)
        HfModel.set_log_model_output(True)
        HfChatModel.set_log_model_input(False)
        HfChatModel.set_log_model_output(True)

    # others
    model_verbose = True
    verbose = True
    print_answer_for_each_example = True
    num_shot = 4

    # run id
    run_id = f"{dataset_name}_{reasoning_method}" if not is_running_in_jupyter() else f"test_{dataset_name}_{reasoning_method}"
    if add_continuation:
        run_id += "_continuous"
        if bn_method:
            run_id += f"_bn{bn_method[0]}"
        if reward_alpha is not None:
            run_id += f"_rm"

    run_id += f"_nAct{n_actions}_depth{depth_limit}"
    if not terminate_on_terminal_node:
        run_id += "F" # indicate forcing the iterations to be "Finished"
    if is_mcts_method(reasoning_method):
        run_id += f"_nIter{n_iters}"
    if is_mcts_method(reasoning_method) and roll_out_steps < 5000:
        run_id += f"_m{roll_out_steps}"

    if use_critic:
        run_id += "_critic"
        
    if check_action_sim:
        run_id += "_noSim"

    if 'binary_sampling' in terminate_constraints:
        if 'verify' in terminate_constraints:
            run_id += f"_TermV{str(sample_size_terminate)}"
        else:
            run_id += f"_Term{str(sample_size_terminate)}"
        run_id += f'{str(sample_threshold_terminate).replace(".", "")}'

    if 'reward_threshold' in terminate_constraints:
        assert r_terminating is not None
        code = str(r_terminating).replace('.', '')
        if terminate_ORM_name:
            run_id +=  f'ORM{code}' if '_Term' in run_id else f"_TermORM{code}"
        else:
            run_id +=  code if '_Term' in run_id else f"_Term{code}"
        
    else:
        if r_terminating:
            run_id += f"_SearchTerm{str(r_terminating).replace('.', '')}"

    # create root dir
    root_dir = f"{model_name.split('/')[-1]}_results/"
    if eval_model_name != model_name:
        root_dir += f"{eval_model_name.split('/')[-1]}/"
    root_dir += f"{run_id}"
    if bn_model_name == "Qwen/Qwen3-32B-AWQ" and model_name != "Qwen/Qwen3-32B-AWQ":
        root_dir += "_bn_qwen"
    if eval_idx:
        root_dir += f"_eval{eval_idx[0]}-{eval_idx[-1]}"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)

    # save config to root dir
    save_config_path = os.path.join(root_dir, f"{reasoning_method}_config.json")
    with open(save_config_path, "w") as f:
        json.dump(search_config.to_dict(), f, indent=4)

   # load models
    if reasoning_method == "rap":
        assert model_name == "meta-llama/Meta-Llama-3-8B", f"Rap only support meta-llama/Meta-Llama-3-8B, but got {model_name}"
        assert eval_model_name == "meta-llama/Meta-Llama-3-8B", f"Rap only support meta-llama/Meta-Llama-3-8B, but got {eval_model_name}"
        base_model = HfModel.load_from_hf(model_name, device=device, verbose=model_verbose)
        eval_base_model = HfModel.load_from_hf(eval_model_name, device=device, verbose=model_verbose)
    elif reasoning_method == "rest" or "bfs":
        assert model_name != "meta-llama/Meta-Llama-3-8B", f"Rest does not support meta-llama/Meta-Llama-3-8B"
        assert eval_model_name != "meta-llama/Meta-Llama-3-8B", f"Rest does not support meta-llama/Meta-Llama-3-8B"
        base_model = HfChatModel.load_from_hf(model_name, device=device, enable_thinking=search_config.enable_think_policy, sys_prompt=None, verbose=model_verbose)
        eval_base_model = HfChatModel.load_from_hf(eval_model_name, device=device, enable_thinking=search_config.enable_think_eval, sys_prompt=None, verbose=model_verbose)
        if terminal_gen_model_name:
            terminal_model = HfChatModel.load_from_hf(terminal_gen_model_name, device=device, enable_thinking=search_config.enable_think_terminal_gen, sys_prompt=None, verbose=model_verbose)
        if 'reward_threshold' in terminate_constraints and terminate_ORM_name:
            terminate_ORM = HfChatModel.load_from_hf(terminate_ORM_name, device={"": 1}, enable_thinking=True, sys_prompt=None, verbose=model_verbose)
    else:
        raise ValueError(f"Unknown inference method: {reasoning_method}") 

    # setup logging (one for mcts process, one for inference costs)
    logger = setup_logging(run_id, root_dir, add_console_handler=True, verbose=verbose)
    inference_logger = InferenceLogger(run_id='', root_dir=root_dir, override=override_log_result)
    base_model.inference_logger = inference_logger
    eval_base_model.inference_logger = inference_logger
        
    if terminal_gen_model_name:
        terminal_model.inference_logger = inference_logger
    if terminate_ORM_name:
        terminate_ORM.inference_logger = inference_logger

    # load world model and policy evaluator
    if reasoning_method == "rap":
        # load prompt dict
        useful_prompt_dict = gsm8k_rap['evaluator']
        prompt_dict = gsm8k_rap['actor_dynamics']
        system_message_dict = {k: prompt_dict[k] for k in ["instruction", "interactive_examples", "useful_examples"]}
        task_instruction = get_task_instruction_with_examples(system_message_dict, num_shot)
        user_message_dict = {k: prompt_dict[k] for k in ['question_prefix', 'subquestion_prefix', 'overall_question_prefix', 'answer_prefix']}

        # load world model and policy evaluator
        world_model = RAPWorldModel(
            base_model=base_model, 
            task_instruction=task_instruction, 
            usr_msg_dict=user_message_dict, 
            max_length=max_length, 
            batch_size=1, 
            n_confidence=n_confidence,
        )
        policy = RAPPolicy(
            base_model=base_model, 
            task_instruction=task_instruction, 
            usr_msg_dict=user_message_dict, 
            n_actions=n_actions, 
            temperature=0.8,
            force_terminating_on_depth_limit=force_terminating_on_depth_limit, 
            depth_limit=depth_limit, 
            dataset_name=dataset_name, 
            max_length=max_length
        )
        evaluator = RAPEvaluator(
            base_model=eval_base_model, 
            useful_prompt_dict=useful_prompt_dict, 
            temperature=0.8, 
            reward_alpha=0.5, 
            reward_confidence_default=0.8, 
            max_length=max_length
        )
        world_model.n_shots = num_shot
        policy.n_shots = num_shot
    elif reasoning_method == "rest" or "bfs":
        eval_instruction = qa_rest['evaluator']
        task_instruction = get_task_instruction(qa_rest,method="rest")
        
        world_model = RestWorldModel(
            base_model=terminal_model if terminal_gen_model_name else base_model, 
            # terminate constraints
            terminate_constraints=terminate_constraints, 
            r_terminating=r_terminating, 
            sample_size_terminate=sample_size_terminate, 
            sample_threshold_terminate=sample_threshold_terminate, 
            # llm parameter
            max_length=max_length
        )
        policy = RestPolicy(
            base_model=base_model, 
            task_instruction=task_instruction, 
            n_actions=n_actions, 
            temperature=0.7, 
            force_terminating_on_depth_limit=force_terminating_on_depth_limit, 
            depth_limit=depth_limit, 
            dataset_name=dataset_name, 
            max_length=max_length, 
            check_action_sim=check_action_sim
        )

        if eval_model_name ==  "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
            evaluator = RestEvaluator2(base_model=eval_base_model)
        else:
            assert eval_model_name == "Qwen/Qwen3-32B-AWQ" or eval_model_name.startswith("meta-llama/Meta-Llama-3-8B")
            evaluator = RestEvaluator(base_model=eval_base_model, eval_instruction=eval_instruction, save_dir=root_dir, 
                think_for_correctness=think_for_correctness, think_for_usefulness=think_for_usefulness,
                n_for_correctness = n_for_correctness,
                n_for_usefulness = n_for_usefulness,)

    # bn evaluator
    bn_evaluator = None
    if bn_method:
        if bn_model_name:
            bn_model= HfChatModel.load_from_hf(bn_model_name, device=device, enable_thinking=search_config.enable_think_policy, sys_prompt=None, verbose=model_verbose)
            bn_model.inference_logger = inference_logger
        else:
            assert isinstance(base_model, HfChatModel), f"BN model must be a HfChatModel, got {type(base_model)}"
            bn_model = base_model
        
        bn_evaluator = BNEvaluator(base_model=bn_model, method=reasoning_method, max_length=max_length, max_new_tokens_for_bn_eval=max_new_tokens_for_bn_eval, max_try_for_bn_eval=max_try_for_bn_eval, eval_method=bn_method)

    # for saving results
    result_saver = None
    if is_mcts_method(reasoning_method):
        result_saver = TreeToJsonl(run_id='', root_dir=root_dir, override=override_log_result)
        result_saver_unselected = TreeToJsonl(run_id='unselected_simulate', root_dir=root_dir, override=override_log_result)
    else:
        result_saver = ResultDictToJsonl(run_id='', root_dir=root_dir, override=override_log_result)
        bucket_saver = ResultDictToJsonl(run_id='bucket', root_dir=root_dir, override=override_log_result)
    existing_results = result_saver.results
    assert len(existing_results) == 0, f"No support for using existing results from v0.1.4"

    # run
    example_idx = 0
    full_dataset = load_qa_dataset(dataset_name)

    if head_size_eval is not None:
        full_dataset = full_dataset[:head_size_eval]
    retrieve_answer = make_retrieve_answer(base_model)

    begin_time = time.time()
    for example in tqdm(full_dataset):
        if eval_idx and example_idx not in eval_idx:
            logger.debug(f"Skipping example {example_idx}")
            example_idx += 1
            continue
        question = example["question"]

        if is_mcts_method(reasoning_method):
            algo_output = mcts(question, example_idx, search_config, world_model, policy, evaluator, bn_evaluator)
            paths_to_save = [algo_output.trace_of_nodes] + algo_output.trace_in_each_iter
            # add "The answer is ..." to the terminal states
            for path in paths_to_save:
                if len(path) > 0 and hasattr(path[-1], 'is_terminal') and path[-1].is_terminal:
                    _ = retrieve_answer(path[-1].state, question)
            # get_result_from_mcts(algo_output.root, question, retrieve_answer, weight_policy='edge')
            result_saver.append_result(paths_to_save)
            result_saver_unselected.append_result(algo_output.unselected_terminal_paths_during_simulate)
            answer_pred = retrieve_answer(paths_to_save[0][-1].state, question) # Final answer extracted via DFS            
        elif reasoning_method == "bfs":
            vote_answers, answer_reward_d, buckets = bfs_topk(question, example_idx, search_config, world_model, policy, evaluator, retrieve_answer, bn_evaluator, return_buckets=True)
            result_saver.append_result(result={'answer_vote': vote_answers, 'answer_reward': answer_reward_d, 'truth': example['answer']})
            bucket_saver.append_result(result= {k:[node.to_dict() for node in vs] for k,vs in dict(buckets).items() })
            if len(vote_answers) > 0:
                answer_pred = max(vote_answers, key=lambda answer: vote_answers[answer])
            else:
                answer_pred = ''
        if print_answer_for_each_example:
            logger.debug(f"Prediction: {answer_pred}; Truth: {example['answer']}\n\n\n")   

        example_idx += 1
    end_time = time.time()
    logger.info(f"Total time: {end_time - begin_time}")
    for role_prefix in VALID_ROLES_PREFIX:
        logger.info(f"{role_prefix}: \t {str(base_model.inference_logger.get_metrics_by_prefix(role_prefix))}")

    if result_saver and is_mcts_method(reasoning_method):
        metric = get_accuracy(full_dataset, results_from_file=result_saver)
        logger.info(f"Run ID: {run_id}, Accuracy: {metric['accuracy']:.4f}, Correct Count: {metric['correct_count']}, Total Examples: {metric['total_examples']}")


# llm
model_name =  "meta-llama/Meta-Llama-3-8B-Instruct" # "Qwen/Qwen3-32B-AWQ"  #"meta-llama/Meta-Llama-3-8B" #
eval_model_name =   "meta-llama/Meta-Llama-3-8B-Instruct" #"RLHFlow/Llama3.1-8B-PRM-Deepseek-Data" #"Qwen/Qwen3-32B-AWQ"   #"meta-llama/Meta-Llama-3-8B" # 
bn_model_name = None # "Qwen/Qwen3-32B-AWQ" 
eval_idx = list(range(20, 36))
main(
    dataset_name="math500", 
    model_name=model_name,
    eval_model_name=eval_model_name, 
    reasoning_method="bfs", 
    add_continuation=False, 
    bn_method=None, 
    bn_model_name=bn_model_name, 
    eval_idx=eval_idx
)
# main(
#     dataset_name="math500", 
#     model_name=model_name,
#     eval_model_name=eval_model_name, 
#     reasoning_method="rap", 
#     add_continuation=False, 
#     bn_method=None, 
#     bn_model_name=bn_model_name, 
#     eval_idx=eval_idx
# )
# main(
#     dataset_name="math500", 
#     model_name=model_name,
#     eval_model_name=eval_model_name, 
#     reasoning_method="rap", 
#     add_continuation=True, 
#     bn_method="direct", 
#     bn_model_name=bn_model_name, 
#     eval_idx=eval_idx
# )
