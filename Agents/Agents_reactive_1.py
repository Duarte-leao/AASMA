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
            # time.sleep(0.3)
            environment.render()
            
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)

        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b


class Reactive_1(Agent):

    def __init__(self, n_robots, n_actions):
        
        super(Reactive_1, self).__init__(f"Reactive Agent 1")
        self.n_agents = n_robots
        self.n_actions = n_actions
        self.weight_agents_nearby = 0.5
        self.weight_distance = 1
        self.observation = None 
    

    def action(self, observation, num_dir1) -> int:

        self.observation = observation
        agents_positions = self.observation[:self.n_agents, :]
        dirt_position1 = self.observation[self.n_agents:num_dir1+self.n_agents, :]
        dirt_position2 = self.observation[num_dir1+self.n_agents:, :]

        actions = []
        heuristic_dirt1 = math.inf
        heuristic_dirt2 = math.inf
        agents_nearby_dirt2 = 0
        
        for i in range(self.n_agents):
            
            if len(dirt_position1) > 0:
                dirt1 = self.closest_dirt(agents_positions[i], dirt_position1) 
                distance_to_dirt1 = cityblock(agents_positions[i], dirt1)
                heuristic_dirt1 = self.heuristic(distance_to_dirt1, 0)

            if len(dirt_position2) > 0:
                dirt2 = self.closest_dirt(agents_positions[i], dirt_position2) 
                distance_to_dirt2 = cityblock(agents_positions[i], dirt2)
                agents_nearby_dirt2 = self.agent_nearby(agents_positions, dirt2)
                heuristic_dirt2 = self.heuristic(distance_to_dirt2, agents_nearby_dirt2)
           

            if heuristic_dirt1 < heuristic_dirt2:
                if np.all(np.equal(agents_positions[i], dirt1)):
                    action = 5
                    actions.append(action)

                else:
                   actions.append(self.direction_to_go(agents_positions[i], dirt1))

            else:
                if agents_nearby_dirt2 >= 2 and np.all(np.equal(agents_positions[i], dirt2)) :
                    action = 5
                    actions.append(action)

                elif agents_nearby_dirt2 < 2:
                    if len(dirt_position1)>0:
                        actions.append(self.direction_to_go(agents_positions[i], dirt1))

                    else:
                        actions.append(np.random.randint(self.n_actions-1))
                        
                else:
                    actions.append(self.direction_to_go(agents_positions[i], dirt2))
                    
        return actions
    

    def heuristic(self, distance, agents_nearby):
        return self.weight_distance * distance + self.weight_agents_nearby * agents_nearby
    
    def closest_dirt(self, agent_position, dirt_positions):

        min = math.inf
        closest_dirt_position = None
        
        for p in (dirt_positions):

            distance = cityblock(agent_position, p)
            if distance < min:
                min = distance
                closest_dirt_position = p
            
        return closest_dirt_position


    def agent_nearby(self, agents_positions, dirt2):

        nearby = 0
        for i in range(self.n_agents):
            
            distance = cityblock(agents_positions[i], dirt2)
            if distance < 3:
                nearby += 1 
            if nearby == 2:
                break

        return nearby
    

    def direction_to_go(self, agent_position, dirt_position):

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
    n_episodes = 100
    n_actions = 6
    
    agent =  Reactive_1( n_robots, n_actions)

    results_time = {}
    results_battery = {}

    result_t, result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
    results_time[agent.name] = result_t
    results_battery[agent.name] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["yellow"])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["yellow"])




