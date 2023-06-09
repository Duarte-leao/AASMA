from Environment_clean_QL import CleaningEnvQL
from Results_analysis import compare_results
from Agents import Agent
import argparse
import numpy as np
from gym import Env
from scipy.spatial.distance import cityblock
from collections import defaultdict
import itertools    
import cloudpickle
import time
import dill



def run_agent( agents,  n_episodes: int, grid_size, n_robots, max_steps) -> np.ndarray:

    results = np.zeros(n_episodes)
    

    for episode in range(n_episodes):

        environment = CleaningEnvQL(grid_size, n_robots, max_steps)
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()
        observations = tuple(itertools.chain.from_iterable(observations))
        done = False

        while not done:
            
            # environment.render()
            # time.sleep(0.3)
            if (environment.num_dirt1 + environment.num_dirt2) == 0:
                 break
            
            actions = [agent.action(observations, environment.num_dirt1) for agent in agents]
            next_observations, rewards, done, info = environment.step(actions)
            next_observations = tuple(itertools.chain.from_iterable(next_observations))

            for a, agent in enumerate(agents):
                if agent.training:
                    agent.update(observations, actions[a], next_observations, rewards)
        
            observations = next_observations
        
        results[episode] = environment.current_step 
        environment.close()

    return results

    

if __name__ == '__main__':


    n_agents_testing = 1
    agent_id = np.arange(n_agents_testing)
    grid_size = 5
    n_robots = 2
    max_steps = 300
    n_episodes = 20

    with open('Q_learning_hard_3.pkl', 'rb') as file:
        agent = dill.load(file)

    results = {}
    result = run_agent( agent,  n_episodes, grid_size, n_robots, max_steps)
    results[agent[0]] = result


compare_results(results, title="Agents on 'Cleaning' Environment", colors=["orange"])