# import sys
# sys.path.append('../..')
import io
import re
import json
import os
import ast
import numpy as np
from typing import Optional, Union, TypedDict, NamedTuple, List, Dict, Tuple, Any
from langagent.reasoner_base import AlgorithmOutput, WorldModel, Policy, RewardModel
from langagent.search.node import register_type
from langagent.base_llm import DETERMINISTIC_TEMPERATURE, HfChatModel, HfModel, DEFAULT_MAX_LENGTH
from langagent.eval import parse_reasoning_and_label
from langagent.prompt import PromptTemplate
from langagent.langreason.common import eval_output, extract_numerical_answer, strip_num
import random
import copy
import itertools
from collections import defaultdict
import logging
from langagent.framework_config import gsm8k_rap
logger = logging.getLogger(__name__)

def create_role(llm_role, example_idx=None, from_phase=""):
    VALID_LLM_ROLES = ["evaluator_logits_ORM", "dynamics", "dynamics_verify", "dynamics_critic", "evaluator_logits", "evaluator_correctness", "evaluator_usefulness", "policy", "evaluator_logits", "bn_entropy_agg", "bn_entropy_remove", "bn_eval", "bn_entropy", None, ""]
    VALID_PHASES = ['expand', 'continuation', 'simulate', 'sort', '', None]
    assert llm_role in VALID_LLM_ROLES, f"Invalid llm_role: {llm_role}"
    assert from_phase in VALID_PHASES, f"Invalid from_phase: {from_phase}"
    role = llm_role 
    role += f"_{example_idx}" if example_idx is not None and example_idx != '' else ''
    role += f"_{from_phase}" if from_phase is not None and from_phase != '' else ''
    return role
    
PolicyAction = str
@register_type
class RapStep(NamedTuple):
    sub_question: PolicyAction
    sub_answer: str
    confidence: float

    def get_action(self):
        return self.sub_question
    
    def get_answer(self):
        return self.sub_answer

@register_type
class RestStep(NamedTuple):
    action: PolicyAction

    def get_action(self):
        return self.action
    
    def get_answer(self):
        return self.action

QAState = list[Union[RapStep, RestStep]]

def make_retrieve_answer(base_model):
    def retrieve_answer(final_state: list, question: str) -> Optional[str]:
        '''
        final_state should be a world_model.QAState if being a list
        '''
        def add_last_step_to_final_state_and_return_float_answer():
            logger.debug(f"Original last step: {final_state[-1].get_answer()}")
            sample_answers = []
            for _ in range(5): # sample 5 times
                if isinstance(final_state[0], RestStep):
                    solution = verbalize_rest_state(question, final_state)
                elif isinstance(final_state[0], RapStep):
                    solution = verbalize_rap_state(question, final_state)
                answer = extract_numerical_answer(base_model, solution)
                logger.debug("Retrieve answer from the path, and append it to the final state: " + answer)
                sample_answers.append(answer)

            # choose the most common answer
            answer = max(set(sample_answers), key=sample_answers.count)
            logger.debug("Final answer after sampling: " + answer)

            # append the answer as the last step in the final trace
            if isinstance(final_state[0], RestStep):
                final_state.append(RestStep(action="The answer is " + answer))
            elif isinstance(final_state[0], RapStep):
                final_state.append(RapStep(sub_question="What is the final answer?", sub_answer="The answer is " + answer, confidence=1.0))
            else:
                raise ValueError(f"Unknown step type: {type(final_state[0])}")
            logger.debug(f"Added last step: {final_state[-1].get_answer()}")
            return answer

        if final_state is None or len(final_state) == 0:
            return ""
        
        # ensure output is a terminal state
        assert isinstance(final_state, list), f"final_state should be a list, got {type(final_state)}"
        assert isinstance(base_model, (HfChatModel, HfModel, type(None))), f"base_model should be HfChatModel or HfModel, got {type(base_model)}"

        # extract the answer from the last step
        answer = retrieve_answer_from_last_step(final_state[-1])

        # use LLM to infer the answer
        if base_model is not None:
            # if answer is empty, use LLM to infer the answer
            if answer == "":
                answer = add_last_step_to_final_state_and_return_float_answer()
                return answer

            # if answer is not a number, use LLM to infer the answer
            try:
                float(answer)
            except ValueError:
                add_last_step_to_final_state_and_return_float_answer()
        return answer
                
    return retrieve_answer

def retrieve_answer_from_last_step(step) -> Optional[str]:
    
    # extract the answer from the last step
    assert isinstance(step, (RapStep, RestStep, str)), f"step should be RapStep, RestStep or str, got {type(step)}: {step}"
    output = step.get_answer() if isinstance(step, (RapStep, RestStep)) else step
    pattern = r'.*[Tt]he answer is .*?([$.0-9,\-]+)(?:\..*)?'
    match = re.match(pattern, output, re.DOTALL)
    # `re.DOTALL` to make .* match across newline characters
    # .*?  
        # . = any character except newline
        # * = zero or more times
        # ? = non-greedy (matches as few characters as possible
    # [...] = character class (match any single character from this set)
        # \- = escaped hyphen (literal minus sign)
    # (?:...) = non-capturing group (groups for precedence but doesn't create a capture group)
        # \. = literal period (escaped because . normally means "any character")
        # .* = any characters, zero or more times
        # ? = makes the entire group optional (zero or one occurrence)
    if match is None:
        answer = ""
    else:
        answer = match[1].replace(',', '').replace('$', '').replace(' ', '').rstrip('.') # 
        if '=' in answer:
            answer = answer[answer.rindex('=') + 1:]
    return answer

def extract_answer_from_aggregation( paths, use_reward=False, weight_policy: str = 'edge', ):
    assert weight_policy in ['edge']

    # extraction
    final_steps_for_answer_extracion = [path[-1].state[-1]  for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)] # path[-1].state is not None and len(path[-1].state) > 0: expanded node that has not been simulated
    if use_reward:
        rewards = [path[-1].reward for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)]
    else:
        rewards = [path[-1].fast_reward for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)]
    depths = [path[-1].depth for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)]
    assert len(final_steps_for_answer_extracion) == len(rewards) == len(depths)

    # aggregation
    answer_dict = defaultdict(lambda: 0)
    num_terminal = 0
    for final_step, reward, depth in zip(final_steps_for_answer_extracion, rewards, depths):
        print("Final step:", final_step)
        answer = retrieve_answer_from_last_step(final_step) 
        if answer == "":
            continue
        else:
            num_terminal += 1
        if weight_policy == 'edge':
            answer_dict[answer] += reward
        elif weight_policy == 'edge_inverse_depth':
            answer_dict[answer] += reward / depth
    
    if num_terminal > 1:
        print(f"Number of terminal nodes with answers: {num_terminal}")
        print("Answer dict:", answer_dict)

    if len(answer_dict) == 0:
        return ""
    return max(answer_dict, key=lambda answer: answer_dict[answer])

def extract_answer_from_dfs_path(final_trace):
    
    terminal_state = final_trace[-1].state if len(final_trace) > 0 else None
    final_step = terminal_state[-1]
    try:
        print("Final step:", final_step)
        output = retrieve_answer_from_last_step(final_step)
    except IndexError:
        print("IndexError in retrieve_answer_from_last_step. terminal_state: ", terminal_state)
        output = ""
    return output
            
def get_accuracy(full_dataset, results_from_file, extract_method="dfs", verbose=False):
    correct_count = 0
    incorrect_indices = []
    for example_idx in range(len(full_dataset)):
        if verbose:
            print("\nexample_idx: ", example_idx)
        # ground truth
        example = full_dataset[example_idx]
        answer = example['answer']
        
        # model output
        try:
            final_trace_plus_all = results_from_file.results[example_idx] 
        except IndexError:
            print(f"results_from_file.results[{example_idx}] -> IndexError")
            break
        
        # extract answer from model output
        if extract_method == "dfs":
            output = extract_answer_from_dfs_path(final_trace_plus_all[0])
        elif extract_method == "aggregation":
            output = extract_answer_from_aggregation(final_trace_plus_all[1:])
        else:
            raise ValueError(f"Unknown extract_method: {extract_method}")

        # caculate accuracy
        correct = eval_output(answer, output)
                
        if verbose:
            print("answer: ", answer, "; output: ", output, "; correct: ", correct)
        if not correct:
            incorrect_indices.append(example_idx)
            
        correct_count += correct
        example_idx += 1

    accuracy = correct_count / len(full_dataset)
    return {"accuracy": accuracy, "correct_count": correct_count, "total_examples": len(full_dataset), "incorrect_indices": incorrect_indices}

def get_task_instruction(prompt_dict, method="rap"):

    if method == "rap":
        instruction = prompt_dict['instruction']
    elif method == "rest":
        instruction = prompt_dict["policy_sys"]
    return instruction

def get_task_instruction_with_examples(prompt_dict, num_shot=0):
    # prompt dict
    prompt_dict = copy.deepcopy(prompt_dict) 
    prompt_dict['interactive_examples']= random.sample(list(prompt_dict['interactive_examples']), k=num_shot)
    
    # instruction
    with io.StringIO() as f:
        f.write(prompt_dict['instruction'] + '\n\n')
        for idx, example in enumerate(prompt_dict['interactive_examples']):
            f.write(example.format(idx=idx + 1) + '\n\n')

        instruction = f.getvalue()
    
    return instruction
    
class RAPWorldModel(WorldModel):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model,
                 task_instruction,
                 usr_msg_dict,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 max_length=None,
                 max_new_tokens=None,
                 top_k=50,
                 top_p=0.95,
                 early_stop_base=None,
                 early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.n_shots = 4
        self.max_length = max_length
        print("max_length in world model: ", max_length)
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        # self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.task_instruction = task_instruction
        self.usr_msg_dict = usr_msg_dict

    def init_state(self) -> list:
        return []

    def _generate_prompt(self, example: str, state: QAState, action: PolicyAction) -> str:
        
        state = state.copy()
        # system message
        task_instruction = self.task_instruction
        
        # user message
        user_message = verbalize_rap_state(example, state)
        user_message += self.usr_msg_dict["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1) + " " + action + "\n"
        user_message += self.usr_msg_dict["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1)
        
        if isinstance(self.base_model, HfChatModel):
            assert self.n_shots == 0
            self.base_model.sys_prompt = task_instruction
            return user_message
        elif isinstance(self.base_model, HfModel):
            return task_instruction + user_message
        else:
            raise ValueError(f"Unknown model type: {type(self.base_model)}")
        
    def step(self, example: str, state: QAState, action: PolicyAction, example_idx: int=None, from_phase="") -> tuple[QAState, dict]:
        assert from_phase in ["expand", "continuation", "simulate"]
        model_input = self._generate_prompt(example, state, action)
        logger.debug("\n>>>>>>>>> + 1 Dynamics Call; Output (BEGIN) <<<<<<<<<")
        answer_dict = defaultdict(list)
        for start1 in range(0, self.n_confidence, self.batch_size):
            end1 = min(start1 + self.batch_size, self.n_confidence)
            num = end1 - start1
            
            outputs = self.base_model.batch_generate([model_input]*num, role=create_role("dynamics", example_idx, from_phase), temperature=self.temperature, max_length=self.max_length, max_new_tokens=self.max_new_tokens, do_sample=True, top_k=self.top_k, top_p=self.top_p, stop='\n', new_line_stop=True)
                           
            for output in outputs:
                result = output.strip()
                answer = retrieve_answer_from_last_step(result)    
                if answer is None or answer == "":
                    logger.warning(f"No answer found via `retrieve_answer_from_last_step` from {result}")
                    continue           
                answer_dict[answer].append(result)

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = 0, result  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[0]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        new_state = state.copy()
        new_state.append(RapStep(action, answer, confidence))
        aux = {'confidence': confidence}

        # debug info
        logger.debug(f"Selected Answer: {answer}")
        logger.debug(f"Confidence: {confidence}")
        logger.debug(">>>>>>>>> + 1 Dynamics Call; Output (END) <<<<<<<<<\n")
        return new_state, aux

    def is_terminal(self, state: QAState, example=None, fast_reward: float=None, example_idx: int=None, from_phase: str='') -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False

def get_terminal_prompt(from_rest=False):

    if from_rest:
        common_part_from_rest = """Given a science or math problem and a corresponding solution that may be incomplete, your task is to judge whether the solution has already reached a final answer or \
    conclusion for the problem. If the solution has already reached a final answer or conclusion, you should directly output """
        terminate_prompt = common_part_from_rest + """'Yes'. Otherwise, you should directly output 'No'. Output only one word: Yes or No."""
    else:
        terminate_prompt = """You are a strict checker. Given a science or math problem and a proposed solution (which may be partial), 
your task is to decide whether the solution has already produced the **final numerical or categorical answer** 
to the problem. 

- If the final answer is explicitly stated and no further computation or reasoning is required, output exactly 'Yes'. 
- If more steps are required to reach the final answer, output exactly 'No'. 
- Do not explain. Do not add punctuation. Output only one word: Yes or No.
"""
    return terminate_prompt

# ~~~~~~~~~ RestWorldModel ~~~~~~~~~~~~
class RestWorldModel(WorldModel):
    """ World model for ReST 
    State: [Action 1, Action 2, ...]
    Action: Action
    """
    def __init__(self, base_model, terminate_ORM=None, terminate_constraints=['binary_sampling'], r_terminating=0.9, sample_size_terminate=10, sample_threshold_terminate=0.8, sample_threshold_verify=0.9, max_length=None, max_new_tokens=None):
        super().__init__()
        for constraint in terminate_constraints:
            assert constraint in ['binary_sampling', 'reward_threshold', 'verify'], f"Unknown terminate constraint: {constraint}"
            if constraint == 'reward_threshold':
                assert r_terminating is not None, "r_terminating must be provided when using reward_threshold"
        self.terminate_ORM = terminate_ORM
        self.terminate_constraints = terminate_constraints
        self.r_terminating = r_terminating
        self.sample_size_terminate=sample_size_terminate
        self.sample_threshold_verify=sample_threshold_verify
        self.sample_threshold_terminate=sample_threshold_terminate
        self.base_model = base_model
        self.max_length =  max_length
        self.max_new_tokens = max_new_tokens
        self.terminate_prompt = get_terminal_prompt(from_rest=False)
        self.verify_terminate_prompt = """You are a strict checker. Given a science or math problem and a proposed solution (which may be partial), 
your task is to decide whether the solution has **not yet reached one numerical value to directly answer the proposed question** (the reasoning is incomplete), output exactly: INCOMPLETE.
  - This includes cases where the next step(s) are obvious but not explicitly written 
    
Output only one of: COMPLETE, or INCOMPLETE. 
Do not explain anything. Do not add extra text.
"""
        self.critic = """""Given a science or math problem and a corresponding solution that may be incomplete, your task is to give some advice on how to solve the problem based on current steps or what to consider next."""

    def init_state(self) -> list:
        return []

    def step(self, example: str, state: QAState, action: PolicyAction, example_idx: int=None, from_phase="") -> Union[QAState, Tuple[QAState, dict]]:
        new_state = state.copy()
        new_state.append(RestStep(action))
        return new_state, {"confidence": 1.}

    # def is_terminal(self, state: QAState) -> bool:
    #     # return state.V >= 0.9 # as in https://github.com/THUDM/ReST-MCTS/blob/main/MCTS/task.py
    #     if len(state) > 0 and "Now we can answer" in state[-1].action:
    #         return True
    #     else:
    #         return False


    # def get_reward(self, question, existing_steps, role=None):
    #     step_str = question + " " + " ".join(existing_steps)
    #     conversation = [{"content": step_str, "role":"user"},
    #                     {"content":"+", "role":"assistant"}]
    #     input_ids = self.base_model.tokenizer.apply_chat_template(conversation,return_tensors="pt").to("cuda")
    #     logits = self.base_model.get_next_token_logits( prompt=None, candidates=['+', '-'], role=role, input_ids=input_ids, toekn_idx_for_logit=-3)
    #     score = np.exp(logits) / np.sum(np.exp(logits))
    #     score = score[0]
    #     return score
        
    def is_terminal(self, state: QAState, example: str, fast_reward: float=None, example_idx: int=None, from_phase: str='') -> bool:
        
        if "reward_threshold" in self.terminate_constraints:
            
            if self.terminate_ORM:
                outcome_reward = self.get_reward(example, extract_existing_steps(state), role=create_role("evaluator_logits_ORM", example_idx, from_phase))
            else:
                assert fast_reward is not None, "fast_reward must be provided when using reward_threshold"
                outcome_reward = fast_reward
            logger.debug(f"def is_terminal: reward_threshold (outcome_reward: {outcome_reward}; fast_reward: {fast_reward})")
            if outcome_reward < self.r_terminating:
                return False
            
        if "binary_sampling" in self.terminate_constraints:
            # usr msg
            logger.debug(f"def is_terminal: binary_sampling")
            self.base_model.sys_prompt = self.terminate_prompt
            user_message = verbalize_rest_state(example, state) + f"Do the above step(s) already provide the final answer to the question: '{example}'"

            # for critic
            # self.base_model.sys_prompt = self.terminate_prompt_with_critic
            # if allow_critic:
            #     output_text = self.base_model(user_message, role=f"dynamics{example_idx}_{from_phase}", temperature=DETERMINISTIC_TEMPERATURE).text.strip()
            #     output_text = output_text.lower().strip()
            #     if output_text == "yes":
            #         return True, None
            #     else: 
            #         return False, output_text

            # non-critic
            
            answer_samples = self.base_model.sample_binary_output(user_message, sample_size = self.sample_size_terminate, target="yes", contrast="no", max_length=self.max_length, max_new_tokens=self.max_new_tokens, role=create_role("dynamics", example_idx, from_phase))
            terminal_score = answer_samples['yes'] / self.sample_size_terminate
        
            if terminal_score < self.sample_threshold_terminate:  
                return False
        
        if "verify" in self.terminate_constraints:
            logger.debug(f"def is_terminal: verify")
            
            self.base_model.sys_prompt = self.verify_terminate_prompt
            user_message = verbalize_rest_state(example, state)
            answer_samples = self.base_model.sample_binary_output(user_message, sample_size = self.sample_size_terminate, target="complete", contrast="incomplete", max_length=self.max_length, max_new_tokens=self.max_new_tokens, role=create_role("dynamics_verify", example_idx, from_phase))
            complete_score = answer_samples['complete'] / self.sample_size_terminate
            logger.debug(f"Rate for completion: {str(complete_score)}")
            if complete_score < self.sample_threshold_verify:  
                if "binary_sampling" in self.terminate_constraints:
                    logger.debug("The task requires a numeric answer. The next step should take this into account.")
                    assert isinstance(state[-1], RestStep)
                    state[-1] = state[-1]._replace(
                        action=state[-1].get_action() + f"One numerical value is expected to directly answer the proposed question. The next step should take this into account."
                    )
                return False
            
            # Revise non-numeric answers
            # if output_text not in ["INCOMPLETE", "NON-NUMERIC"]:
            #     try:
            #         float(output_text)
            #         return True
            #     except ValueError:
            #         state[-1].action += f"The task requires a numeric answer that can be parsed by `float()` in Python. The next step should take this into account."
            #         return False
            # elif output_text == "NON-NUMERIC":
            #     state[-1].action += f"The task requires a numeric answer. The next step should take this into account."
            #     return False
            # else:
            #     return False

        return True

    def generate_critic(self, state: QAState, example: str, example_idx: int=None, from_phrase:str='') -> bool:
        # usr msg
        example_idx = f"_{example_idx}" if example_idx is not None else ''
        user_message = "Question: " + example + "\n"
        for idx, thought in enumerate(state):
            user_message += "Step " + str(idx + 1) + ": " + thought.action + "\n"
        
        # for critic
        self.base_model.sys_prompt = self.critic
        output_text = self.base_model(user_message, role=create_role("dynamics_critic", example_idx, from_phrase), temperature=DETERMINISTIC_TEMPERATURE, max_new_tokens=1024).text.strip()
        output_text = output_text.lower().strip()
        return output_text

class QAPolicy(Policy):
    def __init__(self,
                 base_model,
                 task_instruction,
                 n_actions=4,
                 max_length=None,
                 max_new_tokens=None,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 dataset_name="gsm8k") -> None:
        super().__init__()
        # base model
        self.base_model = base_model
        self.max_length= max_length
        print("max_length in search config for actor: ", max_length)
        self.max_new_tokens= max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # for policy
        self.n_actions = n_actions
        self.task_instruction = task_instruction
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.dataset_name = dataset_name
        
    
    def get_actions(self, example, state: QAState,  n_actions=None, example_idx=None, critic=None, from_phase="") -> list[PolicyAction]:

        at_depth_limit = self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit
        if n_actions is None:
            n_actions = 1 if at_depth_limit else self.n_actions
        temperature = DETERMINISTIC_TEMPERATURE if at_depth_limit else self.temperature # deterministic outputs 
        outputs = self._get_actions(example, state,  n_actions, temperature, at_depth_limit, example_idx, critic=critic,from_phase=from_phase)

        logger.debug(f"\n>>>>>>>>> + {n_actions} Policy Call; Outputs (BEGIN) <<<<<<<<<")
        logger.debug(f"n_actions: {n_actions}")
        for idx, output in enumerate(outputs):
            logger.debug(f"\t Action {idx}: {output}")
        logger.debug(f">>>>>>>>> + {n_actions} Policy Call; Outputs (END) <<<<<<<<<\n")

        # remove duplications but also guarantee order
        # Why do we guarantee oder? 
        # For the potential extension of the code with torch.distributed in LLaMA, which requires the same order across all processes
        outputs = list(dict.fromkeys(outputs)) 
        return outputs
    
    def _get_actions(self, example, state: QAState, n_actions, temperature, at_depth_limit, example_idx, critic: str=None, from_phase=""):
        raise NotImplementedError("_get_actions is not implemented for QAPolicy")


class QAEvaluator(RewardModel):
    def __init__(self,
                 base_model,
                 max_length=None,
                 max_new_tokens=None,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8) -> None:
        super().__init__()
        # base model
        self.base_model = base_model
        self.max_length= max_length
        print("max_length in search config for actor: ", max_length)
        self.max_new_tokens= max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # for evaluator
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        
    def fast_reward(self, example, example_idx, state: QAState, action: PolicyAction, from_phase="") -> tuple[float, dict]:

        logger.debug("\n>>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (BEGIN) <<<<<<<<<")

        useful_prob = self._fast_reward(example, example_idx, state, action, from_phase=from_phase)

        fast_reward = self.calculate_reward(useful_prob)

        logger.debug(f"fast_reward: {fast_reward}")
        logger.debug(">>>>>>>>> + 1 Fast Reward Evaluator Call; Outputs (END) <<<<<<<<<\n")

        return fast_reward, {'r_useful': float(useful_prob)}

def extract_existing_steps(state: QAState) -> list[str]:
    existing_steps = []
    for idx, thought in enumerate(state):
        assert isinstance(thought.get_action(), str)
        existing_steps.append(thought.get_action())
    return existing_steps

# user message
def verbalize_rap_state(question, state, n_shots=4):
    usr_msg_dict = gsm8k_rap["actor_dynamics"]
    user_message = usr_msg_dict["question_prefix"].format(idx=n_shots + 1, question=question) + "\n"
    for idx, (sub_question, sub_answer, _) in enumerate(state):
        user_message += usr_msg_dict["subquestion_prefix"].format(idx=n_shots + 1, sub_idx=idx + 1) + " " + sub_question + "\n"
        user_message += usr_msg_dict["answer_prefix"].format(idx=n_shots + 1, sub_idx=idx + 1) + " " + sub_answer + "\n"
    return user_message

def verbalize_rest_state(question, state):
    """ The format of the prompt is:
        Problem: ...
        Existing Steps:
        Step 1: ...
        Step 2: ...
        ...
        Step n: ...
    """
    question = question + '?' if not question.endswith('?') else question
    verbalized_state = "Problem: " + question 

    existing_steps = extract_existing_steps(state)
    if len(existing_steps) > 0:
        verbalized_state += "\nExisting Steps:\n"
    else:
        verbalized_state += "\nExisting Steps: None\n"

    for idx, action in enumerate(existing_steps):
        verbalized_state += "Step " + str(idx + 1) + ": " + action + "\n"
    return verbalized_state

class RestEvaluator2(QAEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_alpha = 1 # so that reward == r_useful 
        
    def _fast_reward(self, example, example_idx, state: QAState, action: PolicyAction, from_phase="") -> tuple[float, dict]:
        def get_reward(question, existing_steps, next_step, role=None):
            conversation = []
            # question + existing steps
            for k, step in enumerate(existing_steps):
                if k == 0:
                    conversation.append({"content": question + " " + step, "role":"user"})
                else:
                    conversation.append({"content": step, "role":"user"})

            # next step
            input_ids = self.base_model.tokenizer.apply_chat_template(conversation + [{"content": next_step, "role":"user"}, {"content":"+","role":"assistant"}],return_tensors="pt").to("cuda")
            # print(input_ids)
            # with torch.no_grad():
                # logits = base_model.model(input_ids).logits[:,-3,candidate_tokens] #simple version, the +/- is predicted by the '-3' position
                # score = logits.softmax(dim=-1)[:,0] # 0 means the prob of + (1 mean -)
                # score = score[0].detach().to('cpu', dtype=torch.float32).item()
            logits = self.base_model.get_next_token_logits( prompt=None, candidates=['+', '-'], role=role, input_ids=input_ids, toekn_idx_for_logit=-3)
            score = np.exp(logits) / np.sum(np.exp(logits))
            score = score[0]
            return score
        score = get_reward(example, extract_existing_steps(state), action, role=create_role("evaluator_logits", example_idx, from_phase))
        return score
    
    def calculate_reward(self, useful_prob: float) -> tuple[float, dict]:
        """ Same as RestEvaluator.reward. But maintain it for the calling from QAEvaluator.fast_reward """    
        return useful_prob
    
    def reward(self, state: QAState, action: PolicyAction,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        
        return r_useful #, {'r_useful': r_useful, 'r_conf': 1}
    
class RestEvaluator(QAEvaluator):
    def __init__(self, **kwargs):
        task_instruction = kwargs.pop('eval_instruction', None)
        
        
        self.think_for_correctness = kwargs.pop('think_for_correctness', True)
        self.think_for_usefulness = kwargs.pop('think_for_usefulness', True)
        if self.think_for_correctness:
            self.correctness_instruction = task_instruction['correctness_cot']
        else:
            self.correctness_instruction = task_instruction['correctness']
        if self.think_for_usefulness:
            self.usefulness_instruction = task_instruction['usefulness_cot']
        else:
            self.usefulness_instruction = task_instruction['usefulness']
        self.n_for_correctness = kwargs.pop('n_for_correctness', 5)
        self.n_for_usefulness = kwargs.pop('n_for_usefulness', 5)
        self.save_dir = kwargs.pop('save_dir', None)
        
        self.file_path_correctness = os.path.join(self.save_dir, f"correctness.jsonl")
        self.file_path_usefulness = os.path.join(self.save_dir, f"usefulness.jsonl")

        super().__init__(**kwargs)
        self.reward_alpha = 1 # so that reward == r_useful 
        
        
    def _generate_usr_msg(self, example, state: QAState, action: PolicyAction) -> str:
        user_message = verbalize_rest_state(example, state)
        user_message += "New Step to be evaluated: " + action + "\n"

        return user_message
        
    def _fast_reward(self, example, example_idx, state: QAState, action: PolicyAction, from_phase="") -> tuple[float, dict]:
        # def get_value(prompt_answer, temperature, max_length, low, high):
        #     cnt = 2
        #     value = low
        #     while cnt:
        #         try:
        #             value = get_local_value(prompt_answer, value_model, value_tokenizer, max_length=max_length, low=low, high=high)
        #             break
        #         except Exception as e:
        #             print(f'obtain<{method}>score fail!\nError:{e}\n')
        #             cnt -= 1
        #     return value
        # prompt_answer = 'Problem: ' + example + '\nSolution:\n' + str(state)

        # value = get_value(prompt_answer, self.temperature, self.max_length, 0, 1)
        # return value
        
        if isinstance(self.base_model, HfChatModel):
            user_message = self._generate_usr_msg(example, state, action)
        else:
            raise ValueError(f"ReST evaluator only supports HfChatModel, got {type(self.base_model)}")

        def save_results(file_path, score, reasoning, full_output):
            save_item = {
                "example_idx": example_idx, 
                "example": example, 
                "steps": [thought.action for idx, thought in enumerate(state)] + [action], 
                "from_phase": from_phase, 
                "score": score,
                "reasoning": reasoning,
                "text": full_output
            }
            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(save_item, f)
                f.write("\n")

        def generate_score(role, user_message, enable_thinking = True, max_try=4):
            msg = copy.deepcopy(user_message)
            sampled_scores = []
            score_type = "correctness" if "correctness" in role else "usefulness"
            n_sample = self.n_for_correctness if "correctness" in role else self.n_for_usefulness

            logger.debug(f"===== Sample {n_sample} {score_type} scores (Begin) ======")
            for i in range(n_sample):
                logger.debug(f">>>>> Sample {i+1}/{n_sample} <<<<<")
                
                try:
                    output = self.base_model(msg, role=role, max_new_tokens=500, skip_special_tokens= True, enable_thinking=enable_thinking).text
                    if enable_thinking:
                        result_dict = parse_reasoning_and_label(output) # to make sure the output is parsed correctly
                        reasoning = result_dict['reasoning'] if result_dict['reasoning'] is not None else ''
                        full_output = result_dict['text'].replace("`", "").replace("'", "").replace('"', "") if 'text' in result_dict else ''
                        
                        score = float(strip_num(result_dict['label']))
                    else:
                        output = strip_num(output)
                        score = float(output)

                    if "correctness" in role and (score == 0 or score == 1):
                        save_results(self.file_path_correctness, score, reasoning, full_output)
                    elif "usefulness" in role and score >=0 and score <= 1:
                        save_results(self.file_path_usefulness, score, reasoning, full_output)
                    else:
                        float("sds")
                        logger.warning(f"{score_type} Score {score} is out of range.")
                except Exception as e:
                    if enable_thinking:
                        txt = "(REASONING) "+ reasoning if reasoning else  "(REASONING) None; (FULL OUTPUT) "+ full_output
                        logger.warning(f"Error ({e}) in parsing label from output: {txt}.")
                        
                        txt_no_prefix = reasoning if reasoning else full_output
                        msg += f"DONOT follow system message of letting you think, since you have already given some reasoning: {txt_no_prefix}. You MUST STOP reasoning and DIRECTLY output a final score."
                        enable_thinking = False
                    logits = self.base_model.get_next_token_logits(msg, ["1", "0"], role=create_role("evaluator_logits", example_idx, from_phase))
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    score = float(probs[0])
                    if score_type == "correctness":
                        score = 1 if score > 0.6 else 0
                    logger.debug(f"Logit {score_type} score: {score}")
                    logger.debug(e)
                
                logger.debug(f"Parsed {score_type} score: {score}")
                sampled_scores.append(score)
                    
                if "correctness" in role and score == 0:
                    logger.debug(f"===== Sample {n_sample} {score_type} scores (END) ======")
                    return 0
            logger.debug(f"Sampled {n_sample} {score_type} scores: {sampled_scores}")
            logger.debug(f"===== Sample {n_sample} {score_type} scores (END) ======")
            assert len(sampled_scores) == n_sample
            return float(np.mean(sampled_scores))
        
        self.base_model.sys_prompt = self.correctness_instruction
        correctness_score = generate_score(create_role("evaluator_correctness", example_idx, from_phase), user_message, enable_thinking=self.think_for_correctness)
        
        if correctness_score == 0:
            return 0
        else:
            self.base_model.sys_prompt = self.usefulness_instruction
            usefulness_score = generate_score(create_role("evaluator_usefulness", example_idx, from_phase), user_message, enable_thinking=self.think_for_usefulness)
            return usefulness_score
        
        # logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"], role=f"evaluator_logits_{example_idx}_{from_phase}")
        
    
    def calculate_reward(self, useful_prob: float) -> tuple[float, dict]:
        """ Same as RestEvaluator.reward. But maintain it for the calling from QAEvaluator.fast_reward """    
        return useful_prob
    
    def reward(self, state: QAState, action: PolicyAction,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        
        return r_useful #, {'r_useful': r_useful, 'r_conf': 1}

class RestPolicy(QAPolicy):
    def __init__(self, **kwargs):
        self.check_action_sim = kwargs.pop('check_action_sim', False)
        super().__init__(**kwargs)

    def _generate_msg(self, example, state: QAState, critic: str = None, at_depth_limit=False) -> str:

        user_message = verbalize_rest_state(example, state)
        if critic:
            user_message += "Advice: " + critic + "\n"
        
        user_message += "Step " + str(len(state) + 1) + ": "
        if at_depth_limit:
            user_message += "This is the last step, and the answer to the question has to be reached. "
        self.base_model.sys_prompt = self.task_instruction
        return user_message

    def _check_sim(self, embedding, exist_embeddings):
        for j, exist_embedding in enumerate(exist_embeddings):
            similarity = (embedding * exist_embedding).sum(dim=-1)
            
            if similarity > 0.98: # 0.95 is the threshold for similarity
                return True, j, similarity
            
        return False, -1, 0

    def _get_actions(self, example, state: QAState,  n_actions, temperature, at_depth_limit, example_idx, critic: str=None, from_phase="") -> list[PolicyAction]:
        """ 
        Args:
            at_depth_limit: This is for RAP, not REST
        """
        assert isinstance(example_idx, int), f"example_idx should be an integer, got {example_idx}"
        # print(f"example_idx: {example_idx}")
        if isinstance(self.base_model, HfChatModel):
            txt_or_msg = self._generate_msg(example, state, critic=critic, at_depth_limit=at_depth_limit)
        else:
            txt_or_msg = self.task_instruction + self._generate_msg(example, state, critic=critic, at_depth_limit=at_depth_limit)

        outputs = []
        embeddings = []
        existing_steps = extract_existing_steps(state)
        for idx in range(0, n_actions):
            n_retry_repeat = 0
            while True:
                if self.check_action_sim:
                    assert len(embeddings) == len(outputs), "embeddings and outputs should have the same length"
                    # print(txt_or_msg    )
                    output_text, embedding = self.base_model(txt_or_msg, role=create_role("policy", example_idx, from_phase), temperature=temperature, \
                        max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, \
                        new_sent_stop=False, enable_thinking=False, return_embedding=self.check_action_sim)
                    # print(output_text.text.strip())
                    # print("embedding:", embedding)
                    output_text = output_text.text.strip() # enable_thinking=False to make generated action stop correctly using new_sent_stop
                    high_sim, high_sim_idx, similarity = self._check_sim(embedding, embeddings)
                    if high_sim:
                        logger.debug(f"!!!!!!!!!! Found similar embedding (Begin) !!!!!!!!!!")
                        logger.debug(f"Existing text: {outputs[high_sim_idx]}")
                        logger.debug(f"New text: {output_text}")
                        logger.debug(f"Similarity: {similarity}")
                        logger.debug("!!!!!!!!!! Found similar embedding (End) !!!!!!!!!!")
                        continue
                else:
                    output_text = self.base_model(txt_or_msg, role=create_role("policy", example_idx, from_phase), temperature=temperature, \
                        max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, \
                        new_sent_stop=False, enable_thinking=False, return_embedding=False).text.strip()
                # output_dict = parse_reasoning_and_label(output_text) # use when enable_thinking=True BUT I need to figure output how to stop correctly
                # output_text = output_dict['label'] if output_dict['label'] is not None else ''
                
                tokens = self.base_model.tokenizer(output_text, return_tensors="pt", add_special_tokens=False).input_ids
                if tokens.shape[1] > 1000:
                    logger.debug(f"!!!!!!!!!! Output is larger than 1000 tokens (Begin) !!!!!!!!!!")
                    logger.debug(f"Output (temperature: {temperature}): {output_text}")
                    logger.debug("\nWith system prompt: ")
                    logger.debug(self.task_instruction)
                    logger.debug("\nWith user prompt: ")
                    logger.debug(txt_or_msg)
                    logger.debug("!!!!!!!!!! Output is larger than 1000 tokens (End) !!!!!!!!!!")
                    continue
                    
                # check whether the prefix is "Next step: "
                if output_text.startswith("Next step: "):
                    output_text = output_text[11:]

                # check whether the prefix is "Step #" where # is the number of existing steps
                if re.match("Step \d+:", output_text) is not None: 
                    matched_text = re.match("Step \d+:", output_text)[0]
                    output_text = output_text[len(matched_text):].strip()
                
                
                # check whether example or any of previous step(s) is in output_text
                if output_text not in existing_steps and example not in output_text:
                    break
                else:
                    logger.debug(f"!!!!!!!!!! Output is in existing steps (Begin) !!!!!!!!!!")
                    logger.debug(f"Output (temperature: {temperature}): {output_text}")
                    logger.debug("\nWith system prompt: ")
                    logger.debug(self.task_instruction)
                    logger.debug("\nWith user prompt: ")
                    logger.debug(txt_or_msg)
                    logger.debug("!!!!!!!!!! Output is in existing steps (End) !!!!!!!!!!")
                    n_retry_repeat += 1
                    if n_retry_repeat > 2:
                        output_text = "ALWAY REPEAT. TERMINATE"
                        break
            outputs.append(output_text)
            if self.check_action_sim:
                embeddings.append(embedding)

        return outputs

class RAPPolicy(QAPolicy):
    def __init__(self, **kwargs):
        self.force_overall_prompt_on_overall_question = kwargs.pop('force_overall_prompt_on_overall_question', True)
        self.force_overall_question_on_overall_prompt = kwargs.pop('force_overall_question_on_overall_prompt', True)
        self.usr_msg_dict = kwargs.pop('usr_msg_dict', {})
        super().__init__(**kwargs)
        self.n_shots = 4 # actor


    # ================== Actor ==================
    def _generate_prompt(self, question, state: QAState, at_depth_limit: bool) -> str:
        
        task_instruction = self.task_instruction

        user_message = verbalize_rap_state(question, state)
        user_message += self.usr_msg_dict["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1)
        if at_depth_limit:
            user_message += " " + self.usr_msg_dict["overall_question_prefix"]
        
        if isinstance(self.base_model, HfChatModel):
            assert self.n_shots == 0
            self.base_model.sys_prompt = task_instruction
            return user_message
        elif isinstance(self.base_model, HfModel):
            return task_instruction + user_message
        else:
            raise ValueError(f"Unknown model type: {type(self.base_model)}")
    
    def _get_actions(self, question, state: QAState, n_actions, temperature, at_depth_limit, example_idx, critic=None, from_phase="") -> list[PolicyAction]:
        assert critic is None, "RAPPolicy does not support critic"
     
        outputs = []
        model_input = self._generate_prompt(question, state, at_depth_limit)
        for idx in range(0, n_actions):
            output_text = self.base_model(model_input, role=create_role("policy", example_idx, from_phase), temperature=temperature, max_length=self.max_length, max_new_tokens=self.max_new_tokens, top_p=self.top_p, stop='\n', new_line_stop=True).text.strip()
            outputs.append(output_text)
        
        if at_depth_limit:
            outputs = [self.usr_msg_dict["overall_question_prefix"] + " " + output for output in outputs]
        
        # if the prefix ( "Now we can answer the question: ") is already there, 
        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            if self.dataset_name == "gsm8k":
                overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$', question)[1]
            else:
                overall_question = question  
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.usr_msg_dict["overall_question_prefix"] in output:
                    logger.debug(f"format it with the original question: {outputs[i]} ")
                    outputs[i] = self.usr_msg_dict["overall_question_prefix"] + ' ' + overall_question
                    logger.debug(f"      -> {outputs[i]}")
                    
        # if actor outputs the original question, format it specifically
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if overall_question.lower() == output.lower(): 
                    outputs[i] = self.usr_msg_dict["overall_question_prefix"] + ' ' + overall_question
        
        return outputs


class RAPEvaluator(QAEvaluator):
    def __init__(self, **kwargs):
        self.useful_prompt_dict = kwargs.pop('useful_prompt_dict', {})
        super().__init__(**kwargs)
        self.n_shot_eval = 4 # evaluator
        
    # ===== Immediate Reward from glm_eval (BEGIN) =====
    def _fast_reward(self, example, example_idx, state: QAState, action: PolicyAction, from_phase="") -> tuple[float, dict]:
        if self.n_shot_eval or isinstance(self.base_model, HfModel):
            with io.StringIO() as f:
                f.write(self.useful_prompt_dict["input"])
                f.write(self.useful_prompt_dict["question_prefix"] + example + "\n")
                for idx, (q, _, _) in enumerate(state):
                    f.write(self.useful_prompt_dict["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.useful_prompt_dict["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
                f.write(self.useful_prompt_dict["useful_prefix"])
                model_input = f.getvalue()
        else:
            assert isinstance(self.base_model, HfChatModel)
            sys_message = "Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. ONLY output one word: 'Yes' or 'No'"
            user_message = "Question 1: " + example + "\n"
            for idx, (q, _, _) in enumerate(state):
                user_message += 'Question 1.{}:'.format(idx + 1) + " " + q + "\n"
            user_message += 'New question 1.{}:'.format(len(state) + 1) + " " + action + "\n"
            user_message += 'Is the new question useful?'

            self.base_model.sys_prompt = sys_message
            model_input = user_message
        
        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"], role=create_role("evaluator_logits", example_idx, from_phase))
        
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]

        logger.debug(f"logits (yes, no): {logits}")
        logger.debug(f"probs (yes, no): {probs}")

        return float(useful_prob)
    
    # ===== Immediate Reward from glm_eval (END) =====

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha) 

    def reward(self, state: QAState, action: PolicyAction,
            r_useful: float = None,
            confidence: float = None) -> tuple[float, dict]:
        # return confidence, {'r_conf': confidence}
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence) # {'r_useful': r_useful, 'r_conf': confidence}

    
# ===== Branching Necessity (BN) Evaluator (BEGIN) =====
sys_prompt_rap = """You are an expert at deciding whether a single reasoning
step is *logically compulsory* given the task and the partial solution path.

────────────────────────────────────────────────────────────────────────
Input fields
(A) Task description - one paragraph.
(B) Partial reasoning path so far - ordered list of sub-questions.
(C) Candidate next step - exactly ONE sub-question describing the next operation.
────────────────────────────────────────────────────────────────────────

ONLY output a single number from 1 to 4.

Scale
4 - **Unavoidable next step**: given the current path, this step must come next to proceed logically.  
3 - Strongly expected: skipping it now would be very unusual, though not impossible.  
2 - Potentially useful but avoidable: alternative coherent next steps exist.  
1 - **Optional**: the step is not logically required at this point.

Think silently, then output the single line - nothing else.
"""

sys_prompt_rest = """You are an expert at deciding whether a single reasoning
step is *logically compulsory* given the task and the partial solution path.

────────────────────────────────────────────────────────────────────────
Input fields
(A) Task description - one paragraph.
(B) Partial reasoning path so far.
(C) Candidate next step.
────────────────────────────────────────────────────────────────────────

ONLY output a single number from 1 to 4.

Scale
4 - **Unavoidable next step**: given the current path, this step must come next to proceed logically.  
3 - Strongly expected: skipping it now would be very unusual, though not impossible.  
2 - Potentially useful but avoidable: alternative coherent next steps exist.  
1 - **Optional**: the step is not logically required at this point.

Think silently, then output the single line - nothing else.
""" # 1/4=> 0.25, 2/4=> 0.5, 3/4=> 0.75, 4/4=> 1.0

usr_prompt_template = PromptTemplate("""(A) {task}
(B) {partial_path}
(C) {candidate_step}""")

import math
from typing import List, Dict
def extract_bne_output(text):
    pattern = r"(\[\s*(?:\{.*?\}\s*,?\s*)+\])"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        extracted = match.group(1)
        return extracted
    return ''

def truncate_clusters(clusters: List[Dict[str, any]], n_candidates: int):
    """
    Select clusters with top counts so that the total count equals n_candidates.
    If adding a cluster would exceed n_candidates, use only the remaining needed count.
    
    Args:
        clusters: list of dicts with {"canonical_action": str, "count": int}
        n_candidates: the number of original candidates (target total)
    
    Returns:
        List[Dict[str, int]]: truncated clusters whose counts sum to n_candidates
    """
    # sort by count descending
    clusters_sorted = sorted(clusters, key=lambda c: c["count"], reverse=True)
    
    result = []
    remaining = n_candidates
    for c in clusters_sorted:
        if remaining <= 0:
            break
        take = min(c["count"], remaining)
        if take > 0:
            result.append({
                "canonical_action": c["canonical_action"],
                "count": take
            })
            remaining -= take
    
    return result

rap_action_desc = """Each action is a single sub-question."""
def cluster_entropy(
    clusters: List[Dict[str, Any]],
    base: float = 2.0,
    normalize: bool = True,
    norm_by: str = "k",  # "k" (recommended) or "N" (only if you truly want that behavior)
) -> Tuple[float, Optional[str]]:
    """
    Compute Shannon entropy over clusters from their counts.
    If normalize=True and norm_by="k", returns Pielou-style normalized entropy in [0,1].

    Returns:
        (entropy_value, best_canonical_action)
    """
    # sanitize counts
    counts = [int(c.get("count", 0)) for c in clusters if int(c.get("count", 0)) > 0]
    total = sum(counts)

    if total == 0:
        return 0.0, None

    # probabilities
    probs = [c / total for c in counts]

    # Shannon entropy
    H = -sum(p * math.log(p, base) for p in probs)

    if not normalize:
        best = max(clusters, key=lambda c: c.get("count", 0)).get("canonical_action")
        return H, best

    if norm_by == "k":
        k = len(counts)
        if k <= 1:
            H_norm = 0.0  # by convention: zero diversity
        else:
            H_norm = H / math.log(k, base)
    elif norm_by == "N":
        # Not recommended: only equals 1 when k == N
        if total <= 1:
            H_norm = 0.0
        else:
            H_norm = H / math.log(total, base)
    else:
        raise ValueError("norm_by must be 'k' or 'N'")

    best = max(clusters, key=lambda c: c.get("count", 0)).get("canonical_action")
    return H_norm, best

def check_overlap_with_context(clusters, base_model, context, example_idx='', is_subquestion=False, max_length=None, max_new_tokens=None):
    n = len(clusters)
    root = list(range(n))

    # --- Step 1: pairwise merge among candidate clusters ---
    for i, j in itertools.combinations(range(n), 2):
        ci, cj = clusters[i], clusters[j]
        if is_subquestion:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two sub-questions {" ("+rap_action_desc+")" if is_subquestion else ""}, decide if they are semantically overlapping given the context."""
        else:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two action descriptions, decide if they are semantically overlapping given the context.

Definition:
- "Overlapping" means the two descriptions express the same underlying operation or 
  one is a specific case/subsumption of the other or have the same effect on the context.
- "Not overlapping" means the operations are mutually exclusive in meaning.

Answer format: return only 'YES' or 'NO' with no punctuation, no explanation.
"""
        user_message = f"""
Context: 
========
{context}
========
New Step A: {ci['canonical_action']}
New Step B: {cj['canonical_action']}
Do these steps express the same underlying operation given the context?
"""
        answer_samples = base_model.sample_binary_output(
            user_message,
            sample_size=3, target="yes", contrast="no", role=create_role("bn_entropy_agg", example_idx),
            max_length=max_length, max_new_tokens=max_new_tokens
        )

        if answer_samples["yes"] > 1:
            # Merge cluster j into cluster i
            ri, rj = root[i], root[j]
            for k in range(n):
                if root[k] == rj:
                    root[k] = ri

    # --- Step 2: aggregate by root ---
    merged = {}
    for idx, cluster in enumerate(clusters):
        r = root[idx]
        if r not in merged:
            merged[r] = {
                "canonical_action": clusters[r]["canonical_action"],
                "count": 0
            }
        merged[r]["count"] += cluster["count"]

    aggregated_clusters = list(merged.values())
    return aggregated_clusters

def check_overlap(clusters, base_model, existing_steps=None, example_idx='', is_subquestion=False):
    """
    Given a list of clusters [{canonical_action, count}], 
    call an LLM to check pairwise semantic overlap.
    Merge overlapping clusters and drop those overlapping with existing steps.
    Returns a merged list of clusters with aggregated counts.
    """

    n = len(clusters)
    root = list(range(n))

    # --- Step 1: pairwise merge among candidate clusters ---
    for i, j in itertools.combinations(range(n), 2):
        ci, cj = clusters[i], clusters[j]
        if is_subquestion:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two canonical action descriptions {" ("+rap_action_desc+")" if is_subquestion else ""}, decide if they are semantically overlapping."""
        else:
            base_model.sys_prompt = f"""You are a strict semantic comparator.
Given two canonical action descriptions, decide if they are semantically overlapping.

Definition:
- "Overlapping" means the two descriptions express the same underlying operation or 
  one is a specific case/subsumption of the other.
- "Not overlapping" means the operations are mutually exclusive in meaning.

Answer format: return only 'YES' or 'NO' with no punctuation, no explanation.
"""
        user_message = f"""
Action A: {ci['canonical_action']}
Action B: {cj['canonical_action']}
Do these overlap semantically?
"""
        answer_samples = base_model.sample_binary_output(
            user_message,
            sample_size=3, target="yes", contrast="no", role=create_role("bn_entropy_agg", example_idx)
        )

        if answer_samples["yes"] > 1:
            # Merge cluster j into cluster i
            ri, rj = root[i], root[j]
            for k in range(n):
                if root[k] == rj:
                    root[k] = ri

    # --- Step 2: aggregate by root ---
    merged = {}
    for idx, cluster in enumerate(clusters):
        r = root[idx]
        if r not in merged:
            merged[r] = {
                "canonical_action": clusters[r]["canonical_action"],
                "count": 0
            }
        merged[r]["count"] += cluster["count"]

    aggregated_clusters = list(merged.values())

    # --- Step 3: remove clusters overlapping with existing steps ---
    if existing_steps:
        filtered = []
        for cluster in aggregated_clusters:
            keep = True
            for step in existing_steps:
                if is_subquestion:
                    base_model.sys_prompt = f"""You are a strict semantic comparator to answer whether the subquestion has been asked before?**

Answer format: return only `YES` or `NO` with no punctuation, no explanation.
"""
                else:
                    base_model.sys_prompt = f"""You are a strict semantic comparator to answer whether the Candidate Action have identical operations and results as the Existing Step, without introducing any extra operations or results?**

Answer format: return only `YES` or `NO` with no punctuation, no explanation.
"""
                user_message = f"""
Existing Step: {step}
Candidate Action: {cluster['canonical_action']}
Do these overlap semantically?
"""
                answer_samples = base_model.sample_binary_output(
                    user_message,
                    sample_size=3, target="yes", contrast="no", role=create_role("bn_entropy_remove", example_idx)
                )
                if answer_samples["yes"] > 1:
                    keep = False
                    break
            if keep:
                filtered.append(cluster)
        return filtered
    return aggregated_clusters


class BNEvaluator:
    "Branching N?"
    def __init__(self, base_model: HfChatModel, method, eval_method="direct", max_length=DEFAULT_MAX_LENGTH, max_new_tokens_for_bn_eval=None, max_try_for_bn_eval=3):
        assert eval_method in ["direct", "entropy", "sc"]
        if method == "rap":
            self._sys_prompt_direct = sys_prompt_rap
        elif method == "rest" or method == "bfs":
            self._sys_prompt_direct = sys_prompt_rest
        else:
            raise ValueError(f"Unknown method: {method}")
        self.base_model = base_model
        self.enable_thinking=False
        self.max_length = max_length
        self.eval_method = eval_method
        self.search_method = method

        self.max_new_tokens_for_bn_eval = max_new_tokens_for_bn_eval
        self.max_try_for_bn_eval = max_try_for_bn_eval

    def _generate_prompt(self, example, state: list[RapStep], action: PolicyAction):
        partial_path = "\n".join([f"{step.get_action()}" for step in state])
        partial_path = "<No Existing Steps>" if partial_path.strip() == "" else partial_path
        candidate_step = action
        model_input = usr_prompt_template.format(task=example, partial_path=partial_path, candidate_step=candidate_step)
        return model_input

    def evaluate(self, example, state: QAState, actions: list[PolicyAction], example_idx: int=None) -> int:
        logger.debug(f">>>>>>>>> BN Evaluation (Begin)  <<<<<<<<<")
        if self.eval_method == "direct":
            assert len(actions) == 1, "direct eval only supports single action"
            bn_score = self.direct_eval(example, state, actions[0], example_idx) # action
        elif self.eval_method == "entropy":
            bn_score = self.entropy_eval(example, state, actions, example_idx) # actions
        elif self.eval_method == "sc":
            bn_score = self.sc_eval(example, state, actions, example_idx) # actions
        else:
            raise ValueError(f"Unknown eval method: {self.eval_method}")
        logger.debug(f"\n Output from BN evaluator: {bn_score}")
        logger.debug(f">>>>>>>>> BN Evaluation (End) <<<<<<<<<")
        return bn_score

    def direct_eval(self, example, state: QAState, action: PolicyAction, example_idx: int=None) -> int:
        model_input = self._generate_prompt(example, state, action)
        self.base_model.sys_prompt = self._sys_prompt_direct
        success_try = False
        for i_try in range(self.max_try_for_bn_eval):
            output = self.base_model(model_input, role=create_role("bn_eval", example_idx), max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval, temperature=0.3, enable_thinking=self.enable_thinking).text.strip()
            try:
                output = int(output)
            except ValueError:
                continue
            if output < 0 or output > 4:
                continue
            success_try = True
            break
        if not success_try:
            return 0
        return output/4

    def sc_eval(self, example, state, actions, example_idx=None, is_subquestion=False):
        # remove empty actions
        actions = [action for action in actions if action.strip() != ""]
        if len(actions) == 1:
            return 1, actions[0]
            
        if self.search_method in ["rest", "bfs"]:
            context = verbalize_rest_state(example, state) 

        else:
            context = verbalize_rap_state(example, state) 
        
        clusters = [ {"canonical_action": action, "count": 1} for action in actions]
        logger.debug(f"\n Input clusters: {clusters}")
        clusters = check_overlap_with_context(clusters, self.base_model, context, example_idx, is_subquestion,  max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval)
        logger.debug(f"\n Output clusters: {clusters}")
        # select the action with the highest count and its proportion
        selected_dict = max(clusters, key=lambda x: x["count"])
        bn_score = selected_dict["count"] / len(actions)
        logger.debug(f"canonical_action: {selected_dict['canonical_action']}")
        return bn_score, selected_dict["canonical_action"]
        

    def entropy_eval(self, example, state, actions, example_idx=None, is_subquestion=False):
        """
        Args:
        actions (list[str]): 
            Example: ["Since 196 is a composite number, we can factor it into its prime factors: 196 = 2^2 × 7 × 7.",
                      "To find the number of positive whole-number divisors of 196, we can start by factoring 196 into its prime factors. We can do this by dividing 196 by the smallest prime number, 2, until it is no longer divisible.",
                      "Since 196 is an even number, it can be expressed as 2 times a smaller number, which is 98. We can factor 98 into 7 times 14, and then further factor 14 into 2 times 7. Therefore, the prime factorization of 196 is 2^2 times 7^2."]
        """
        if len(actions) == 1:
            return 1, actions[0]
            
        if self.search_method in ["rest", "bfs"]:
            self.base_model.sys_prompt =  """You are given a QUESTION and its partial solution (Existing Steps).  
Your task is to group the provided list of candidate next steps (After "List of Candidates for the following step") into clusters.

- Steps that are semantically equivalent must be grouped together.  
- Paraphrase or stylistic differences are irrelevant
- Existing Steps are given only as context and MUST NOT appear in the clusters.  

OUTPUT FORMAT (Python literal list only; must be parsable by ast.literal_eval):  
[
  { "canonical_action": "<CONCRETE calculation(s) and outcome(s) after the Existing Steps>", "count": <the number of the candidates grouped in that cluster> },  
  ...  
]
Rules:
- Each array element represents one cluster.
- No text outside the list.
- The total number of generated words should be no more than 450 words.
"""
            msg = verbalize_rest_state(example, state) + f"""
            List of Candidates for the following step:
            """
            for idx, action in enumerate(actions):
                msg += f"Candidate {idx + 1}: {action}\n" 
        else:
            assert self.search_method == "rap", f"Unknown search method: {self.search_method}"
            self.base_model.sys_prompt =  """You are given a QUESTION and its partial solution (Subquestions which have been answered).  
Your task is to group the provided list of candidate next subquestions (After "List of Candidates for the following step") into clusters.

- Steps that are semantically equivalent must be grouped together.  
- Paraphrase or stylistic differences are irrelevant
- Existing Steps are given only as context and MUST NOT appear in the clusters.  

OUTPUT FORMAT (Python literal list only; must be parsable by ast.literal_eval):  
[
  { "canonical_action": "<a CONCRETE subquestion>", "count": <the number of the candidates grouped in that cluster> },  
  ...  
]
Rules:
- Each array element represents one cluster.
- No text outside the list.
- The total number of generated words should be NO more than 450 words.
"""
            msg = verbalize_rap_state(example, state) + f"""
            List of Candidates for the following step:
            """
            for idx, action in enumerate(actions):
                msg += f"Candidate {idx + 1}: {action}\n" 
        success = 0
        lst_actions_with_counts = []
        for i_try in range(self.max_try_for_bn_eval):
            output = self.base_model(msg, role=create_role("bn_entropy", example_idx), max_length=self.max_length, max_new_tokens=self.max_new_tokens_for_bn_eval, temperature=DETERMINISTIC_TEMPERATURE, enable_thinking=self.enable_thinking).text
            output = extract_bne_output(output)

            try:
                lst_actions_with_counts = ast.literal_eval(output)
                for d in lst_actions_with_counts:
                    if 'canonical_action' not in d or 'count' not in d:
                        continue
            except (SyntaxError, ValueError) as e:
                logger.debug("Invalid JSON:", e)
                continue
            success = 1
        if len(lst_actions_with_counts) == 0 or not success:
            logger.debug(f"No valid output from BN evaluator")
            return 0, None

        existing_steps = extract_existing_steps(state)
        lst_actions_with_counts = check_overlap(lst_actions_with_counts, self.base_model, existing_steps, example_idx=example_idx, is_subquestion=is_subquestion)
        logger.debug(f"clusters after check overlap: {lst_actions_with_counts}")
        lst_actions_with_counts= truncate_clusters(lst_actions_with_counts, len(actions))
        logger.debug(f"clusters after truncate: {lst_actions_with_counts}")
        if lst_actions_with_counts:
            entropy, canonical_action = cluster_entropy(lst_actions_with_counts, base=2, normalize=True, norm_by="k")
            logger.debug(f"entropy: {entropy}")
            logger.debug(f"canonical_action: {canonical_action}")
            return 1-entropy, canonical_action
        else:
            logger.debug("no clusters after filtering")
            return 0, None
# ===== Branching Necessity (BN) Evaluator (END) =====


def test_bn_evaluator():
    evaluator = BNEvaluator(model_name="Qwen/Qwen3-14B", device="cuda")
    evaluator.example = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    
    state = []
    action1 = "How many eggs are left after Janet eats three for breakfast?"
    rap_step1 = RapStep(action1, "", 0)
    state.append(rap_step1)

    # action2 = "How many eggs are left after Janet bakes muffins for her friends?"
    # rap_step2 = RapStep(action2, "", 0)
    # state.append(rap_step2)

    # action = "How many eggs Janet eats tomorrow?"
    action = "How many eggs are left after Janet bakes muffins for her friends?"
    print(evaluator._generate_prompt(state, action))
    print(evaluator.evaluate(state, action))

def test_rap_world_model():
    world_model = RAPWorldModel(base_model=HfChatModel.load_from_hf("Qwen/Qwen3-14B", device="cuda"), max_length=2048)
    world_model.example = """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
    state = []
    action1 = "How many eggs are left after Janet eats three for breakfast?"
    rap_step1 = RapStep(action1, "", 0)
    state.append(rap_step1)
    world_model.step(state, action1)

