from Environment_clean import CleaningEnv
from Results_analysis import compare_results
from Agents import Agent

import numpy as np
import time
import random 
import math
from scipy.spatial.distance import cityblock


def run_agent( agent, n_episodes: int,  grid_size, n_robots1, max_steps)  -> np.ndarray:

    results_t = np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):

        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observation = environment.reset()
        done = False
        while not done:
            time.sleep(0.3)
            environment.render()
            
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)

        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b

class Reactive_2(Agent):

    def __init__(self, n_robots, n_actions):
        super(Reactive_2, self).__init__(f"Reactive Agent 2")
        self.n_agents = n_robots
        self.n_actions = n_actions
        self.observation = None


    def action(self, observation, num_dir1) -> int:

        self.observation = observation
        agents_positions = self.observation[:self.n_agents, :]
        dirt_position1 = self.observation[self.n_agents:num_dir1+self.n_agents, :]
        dirt_position2 = self.observation[num_dir1+self.n_agents:, :]

        actions = []
        for i in range(self.n_agents):

            if len(dirt_position1) == 0:
                
                near_dirt = self.closest_dirt2( agents_positions, dirt_position2, agents_positions[i])
                action = self.direction_to_go1(agents_positions[i], near_dirt)
                

                if np.any(np.all(np.equal(agents_positions[i], dirt_position2), axis=1)) :
                    same_position_robots = np.where(np.all(agents_positions == agents_positions[i], axis=1))[0]
                
                    if len(same_position_robots) > 1:
                        action = 5
                
                actions.append(action)
            else:
                near_dirt = self.closest_dirt1(agents_positions[i], dirt_position1)
                action = self.direction_to_go1(agents_positions[i], near_dirt)
                actions.append(action)
        
        return actions
    

    def direction_to_go1(self, agent_position, dirt_position):

        distances = np.array(dirt_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if abs_distances[0]== 0 and  abs_distances[1]== 0 :
            return 5
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)

        return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    
    def closest_dirt2(self, agent_positions, dirt_postions, atual_agent):

        distance_dirt = np.zeros(len(dirt_postions))
        distance_agents = np.zeros(self.n_agents)
        distance = np.zeros((len(distance_agents), len(distance_dirt)))

        n_dirt = int(len(dirt_postions) / 2)
        
        other_agents = np.delete(agent_positions, np.where(np.all(agent_positions == atual_agent, axis=1)), axis=0)

        if len(other_agents) != len(agent_positions) - 1:
            return self.closest_dirt1( atual_agent, dirt_postions)
        
        for p in range(n_dirt):

            distance_dirt[p] = cityblock(atual_agent, dirt_postions[p])

        for i, agent in enumerate (other_agents):

            distance_agents[i] = cityblock(atual_agent, other_agents[i])

        distance = np.outer(distance_agents, distance_dirt)
        min_index = np.argmin(distance)
        min_position = np.unravel_index(min_index, distance.shape)

        return dirt_postions[min_position[1],:]


    def closest_dirt1(self, agent_position, dirt_positions):

        min = math.inf
        closest_dirt_position = None
        
        for p in (dirt_positions):

            distance = cityblock(agent_position, p)
            if distance < min:
                min = distance
                closest_dirt_position = p
            
        return closest_dirt_position


    def _close_horizontally(self, distances):

        if distances[1] > 0:
            return 4
        elif distances[1] < 0:
            return 3


    def _close_vertically(self, distances):
        if distances[0] > 0:
            return 2
        elif distances[0] < 0:
            return 1


if __name__ == '__main__':

    grid_size = 7
    n_robots = 2
    max_steps = 500
    n_episodes = 20
    n_actions = 6

    agent = Reactive_2(n_robots, n_actions)

    results_time = {}
    results_battery = {}

    result_t, result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
    results_time[agent.name] = result_t
    results_battery[agent.name] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["yellow"])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["yellow"])
