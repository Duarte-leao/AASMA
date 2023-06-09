from Environment_clean import CleaningEnv
from Agents_random import RandomAgent
from Results_analysis import compare_results
from Agents_random import RandomAgent
from Agents_reactive_0 import Reactive_0
from Agents_reactive_1 import Reactive_1
from Agents_reactive_2 import Reactive_2
from Agents_Role import Agent_role


import numpy as np
import time
from scipy.spatial.distance import cityblock

def run_agent( agent, n_episodes: int,  grid_size, n_robots1, max_steps)  -> np.ndarray:

    results_t = np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):

        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observation = environment.reset()
        done = False
        while not done:
        
            environment.render()
            
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)

        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b


if __name__ == '__main__':

    grid_size = 5
    n_robots = 2
    max_steps = 500
    n_episodes = 20

    # 2 - Setup agents
    agents = [
        RandomAgent(n_robots, n_actions=5),
        Reactive_0(n_robots, n_actions=5),
        Reactive_1(n_robots, n_actions=5),
        Reactive_2(n_robots, n_actions=5),
        Agent_role(n_robots, n_actions=5)
    ]

    results_time = {}
    results_battery = {}

    for agent in agents:
        
        result_t,result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
        results_time[agent.name] = result_t
        results_battery[agent.name] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["orange", "green", 'red', 'blue', 'yellow'])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["orange", "green", 'red', 'blue', 'yellow'])

