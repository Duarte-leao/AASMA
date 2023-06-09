from Environment_clean import CleaningEnv
from Results_analysis import compare_results
from Agents import Agent 
import numpy as np
import time


def run_agent( agent, n_episodes: int,  grid_size, n_robots1, max_steps)  -> np.ndarray:

    results_t = np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):

        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observation = environment.reset()
        done = False
        while not done:
            # time.sleep(0.1)
            # environment.render()
            
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)

        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b


class RandomAgent(Agent):

    def __init__(self, n_robots, n_actions):
        
        super(RandomAgent, self).__init__(f"Random Agent")
        self.n_agents = n_robots
        self.n_actions = n_actions

    def action(self, observation, num_dirt) -> int:

        actions = np.random.randint(self.n_actions, size=self.n_agents)
        return actions
    
if __name__ == '__main__':

    grid_size = 5
    n_robots = 2
    max_steps = 200
    n_episodes = 20
    n_actions = 6

    agent = RandomAgent( n_robots, n_actions)
    

    results_time = {}
    results_battery = {}

    result_t, result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
    results_time[agent.name] = result_t
    results_battery[agent.name] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["yellow"])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["yellow"])








