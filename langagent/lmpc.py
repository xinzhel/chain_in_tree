import json
from abc import ABC, abstractmethod
from llmtrack import get_llm

def pre_check(base_model):
    assert base_model in ["Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-14B" , "Qwen/Qwen3-4B"]

class LMPC(ABC):

    def __init__(self, llm_name, config) -> None:
        self.config = config
        self.lmpc_def = config['lmpc_def']
        # llm
        try:
            llm_config =  config['llm']
        except KeyError:
            print("No LLM configuration provided in the config: ", config)
        self.llm = get_llm(llm_name, **llm_config)
        
        # demos
        self.demos_prefix = config['demos_prefix'] if 'demos_prefix' in config else ''
        self.demos_suffix = config['demos_suffix'] if 'demos_suffix' in config else ''
        if "demos" in config:
            self.demos = config['demos']
        elif "demo_path" in config:
            with open(self.config['demo_path'], 'r') as f:
                self.demos = json.load(f)
        else:
            self.demos = ''
        
        # task description
        self.task_desc_trigger = self.config['task_desc_trigger'] if 'task_desc_trigger' in self.config else ''
        
        # generation trigger
        self.gen_trigger = self.config['gen_trigger']  if 'gen_trigger' in self.config else ''
        
    
    @abstractmethod 
    def generate(self, **kwargs):
        raise NotImplementedError("The method `_generate` should be implemented in the subclass.")
    
    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError("The method `run` should be implemented in the subclass.")
        
    

        


