import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):


    def __init__(self, name: str):
        self.name = name
        self.observation = None
        self.training = True

    def see(self, observation: np.ndarray):
        self.observation = observation

    @abstractmethod
    def action(self) -> int:
        raise NotImplementedError()


    def train(self):
        self.training = True

    def eval(self):
        self.training = False