from Environment_clean import CleaningEnv
# from Agents_reactive import run_agent
from Results_analysis import compare_results
import argparse
import numpy as np
from gym import Env
import time


class RandomAgent():

    def __init__(self, agent_id, n_robots, n_actions):
        
        super(RandomAgent, self).__init__
        self.agent_id = agent_id
        self.n_agents = n_robots
        self.n_actions = n_actions

    def action(self, observation) -> int:

        actions = np.random.randint(self.n_actions, size=2)

        return actions
    
if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--episodes", type=int, default=30)
    # opt = parser.parse_args()

    n_agents_testing = 1
    agent_id = np.arange(n_agents_testing)
    grid_size = 5
    n_robots = 2
    max_steps = 300
    

    # 1 - Setup environment
    env = CleaningEnv(grid_size, n_robots, max_steps)

    # 2 - Setup agents
    agents = [
        RandomAgent(agent_id=0, n_robots=2, n_actions=env.n_actions)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        result = run_agent(env, agent, 1)
        results[agent.name] = result

    compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])








