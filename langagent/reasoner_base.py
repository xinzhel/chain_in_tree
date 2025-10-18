from typing import Generic, TypeVar, Union, Protocol, runtime_checkable, Tuple, List, Dict, Any
from abc import ABC, abstractmethod
from tqdm import tqdm

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = Tuple[List[State], List[Action]]


class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, example: str, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...


class Policy(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_actions(self, example, state: State) -> list[Action]: ...

class RewardModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fast_reward(self, example, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace

class Evaluator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    
    def predict_example(self, example, prompt, config_search_algo, world_model, search_config, search_algo, verbose=False,  ):
        # example; dict keys: 'init' (inserted into icl to complete test data), 'goal' (inserted into icl to complete test data), 'plan' (ground truth?), 'question' (The whole test data), 'instance_file' (e.g., "'generated_basic/instance-1.pddl'")
        input_for_prompt = self.input_processor(example)

        # input: example => output: model prediction
        world_model.update_example(input_for_prompt, prompt=prompt)
        search_config.update_example(input_for_prompt, prompt=prompt)
        algo_output = search_algo(config_search_algo, world_model, search_config, verbose=verbose)
        output = self.output_extractor(algo_output)
        
        return output
        
    def evaluate(self, config_search_algo, world_model, search_config, search_algo, shuffle_prompt=True, num_shot=4, verbose=False):
        assert isinstance(self.full_dataset, list), "full_dataset should be a list of examples"
        correct_count = 0
        for example in tqdm(self.full_dataset):
            #  gsm8k_useful_examples.json=>add few_shot prompt; dict keys: 'icl' (task instruction + few-shot demos), 'world_update_pickup', 'world_update_unstack', 'world_update_putdown', 'world_update_stack', 'self-eval'])
            output = self.predict_example(example, config_search_algo, world_model, search_config, search_algo, verbose=verbose, shuffle_prompt=shuffle_prompt, num_shot=num_shot, )
            
            # ground truth
            answer = self.answer_extractor(example)
            
            # caculate accuracy
            correct = self.eval_output(answer, output)
            correct_count += correct
        accuracy = correct_count / len(self.full_dataset)
        return accuracy

    @abstractmethod
    def eval_output(self, answer, output):
        pass
    
