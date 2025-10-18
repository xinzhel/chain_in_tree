from abc import ABC, abstractmethod
from typing import Tuple

class Env(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, bool]:
        """ Take a step in the environment. 
        
        :param action: The action to take in the environment.
        :return: A tuple containing the observation and a boolean indicating if the episode is done.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass