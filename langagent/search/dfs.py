
from ..reasoner_base import State
from typing import NamedTuple, List, Tuple

class DFSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[State]]


