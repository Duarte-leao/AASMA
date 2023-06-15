from Environment_clean import CleaningEnv
from Agents import Agent
import numpy as np
from Results_analysis import compare_results
import time


def run_agent( agent, n_episodes: int,  grid_size, n_robots1, max_steps)  -> np.ndarray:

    results_t = np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):
        
        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observation = environment.reset()
        done = False
        while not done:
            # time.sleep(1)
            # environment.render()
            
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)

        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b


class Reactive_0(Agent):

    def __init__(self, n_robots, n_actions):
        super(Reactive_0, self).__init__(f"Reactive Agent 0")
        self.n_agents = n_robots
        self.n_actions = n_actions
        self.observation = None
 

    def action(self, observation, num_dir1) -> int:
        self.observation = observation
        agents_positions = self.observation[:self.n_agents, :]
        dirt_position1 = self.observation[self.n_agents:num_dir1+self.n_agents, :]
        dirt_position2 = self.observation[num_dir1+self.n_agents:, :]
        actions = self.direction_to_go(agents_positions, dirt_position1, dirt_position2)

        return actions
    
    def direction_to_go(self, pos_agents, pos_dirt1, pos_dirt2):

        actions = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):

            if np.any(np.all(np.equal(pos_agents[i], pos_dirt1), axis=1)):                               
                actions[i] = 5   

            elif np.any(np.all(np.equal(pos_agents[i], pos_dirt2), axis=1)) :
                    
                    same_position_robots = np.where(np.all(pos_agents == pos_agents[i], axis=1))[0]
                    if len(same_position_robots) > 1:
                        actions[i] = 5
                    else:
                        actions[i] = np.random.randint(self.n_actions-1, size=1)
            else:
                actions[i] = np.random.randint(self.n_actions, size=1)

        return actions


if __name__ == '__main__':

    grid_size = 5
    n_robots = 2
    max_steps = 400
    n_episodes = 20
    n_actions = 6

    agent = Reactive_0( n_robots, n_actions)

    results_time = {}
    results_battery = {}

    result_t, result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
    results_time[agent.name] = result_t
    results_battery[agent.name] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["yellow"])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["yellow"])
