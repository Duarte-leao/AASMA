

import lbforaging
from gym.envs.registration import register
import gym 
from gym import spaces
import matplotlib.pyplot as plt
from gym.envs.classic_control import rendering
import time 
import numpy as np
import random


class CleaningEnv(gym.Env):

    def __init__(self, grid_size, num_robots, max_episode_steps):

        super(CleaningEnv, self).__init__()

        self.grid_size = grid_size
        self.num_robots = num_robots
        self.max_episode_steps = max_episode_steps
        
        self.n_actions =  6
        # self.action_space = spaces.Discrete(6)  # 0: No action, 1: Move up, 2: Move down, 3: Move left, 4: Move right, 5: Clean
        # self.observation_space = spaces.MultiBinary((num_robots,2))

        self.grid = np.zeros((grid_size, grid_size))
        self.robot_positions = np.zeros((num_robots, 2), dtype=int)
        self.viewer = None
        self.dirt_removed = 0
        self.num_dirt1 = 0
        self.num_dirt2 = 0
        self.dirt_positions1 ,self.dirt_positions2  = self.generate_dirt()
        self.current_step = 0


    def reset(self):

        # Reset the environment
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.robot_positions = np.zeros((self.num_robots, 2), dtype=int)

        # Set the current step count to zero
        self.current_step = 0

        # Randomly place the robots in the environment
        for i in range(self.num_robots):
            row = np.random.randint(self.grid_size)
            col = np.random.randint(self.grid_size)
            self.robot_positions[i] = [row, col]

        # Return the initial observation
        return self.get_observation()


    def step(self, actions):

        # Execute the specified actions for each robot
        for i in range(self.num_robots):
            action = actions[i]
            robot_pos = self.robot_positions[i]
            
            if action == 1:  # Move up
                robot_pos[0] = max(0, robot_pos[0] - 1)
            elif action == 2:  # Move down
                robot_pos[0] = min(self.grid_size - 1, robot_pos[0] + 1)
            elif action == 3:  # Move left
                robot_pos[1] = max(0, robot_pos[1] - 1)
            elif action == 4:  # Move right
                robot_pos[1] = min(self.grid_size - 1, robot_pos[1] + 1)
            elif action == 5:  # Clean
                if np.any(np.all(np.equal(robot_pos, self.dirt_positions1), axis=1)) :
                    self.dirt_positions1 = np.delete(self.dirt_positions1, np.where(np.all(self.dirt_positions1 == robot_pos, axis=1)), axis=0)
                    self.num_dirt1 = self.num_dirt1 - 1

                if np.any(np.all(np.equal(robot_pos, self.dirt_positions2), axis=1)) :
                    same_position_robots = np.where(np.all(self.robot_positions == robot_pos, axis=1))[0]
                    
                    if len(same_position_robots) > 1:
                        if np.all(actions[same_position_robots] == 5):
                           
                            self.dirt_positions2 = np.delete(self.dirt_positions2, np.where(np.all(self.dirt_positions2 == robot_pos, axis=1)), axis=0)
                            self.num_dirt2 = self.num_dirt2 - 1



        # Increment the step count
        self.current_step += 1

        # Update the environment state
        self._update_environment_state()

        # Calculate the reward
        reward = self._calculate_reward()

        # Check if the episode is done
        done = self.current_step >= self.max_episode_steps

        # Return the updated observation, reward, done flag, and additional info
        return self.get_observation(), reward, done, {}


    def _update_environment_state(self):

        self.grid = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.dirt_positions1.shape[0]):
            self.grid[self.dirt_positions1[i,0], self.dirt_positions1[i,1]] = 1 # Dirt that only one robot can pick up

        for i in range(self.dirt_positions2.shape[0]):
            self.grid[self.dirt_positions2[i,0], self.dirt_positions2[i,1]] = 2  # Dirt that two robots can pick up
                
        for i in range(self.num_robots):
            row, col = self.robot_positions[i]
            self.grid[row, col] = 0.5
        return 
    

    # Should chage it ?? Create a better reward function ??? (not for right now)
    def _calculate_reward(self):
  
        cleaned_area1 = np.sum(self.grid == 1)
        cleaned_area2 = np.sum(self.grid == 2)
        total_area = self.grid_size * self.grid_size
        return 1 - ((cleaned_area1 + cleaned_area2) / total_area)
    

    def get_observation(self):

        # Get the current observation. THIS OBSRVATION HAS THE ROBOT POSITION AND DIRT POSITION [[agent1_pos], [agent2_pos], [dirt1],... ,[dirt n]]

        observation = np.zeros((self.num_robots + self.num_dirt1 + self.num_dirt2, 2 ), dtype=int)  
        for i in range(self.num_robots):
            observation[i] = self.robot_positions[i]

        for i in range(self.num_robots, (self.num_robots + self.num_dirt1)):
            observation[i] = self.dirt_positions1[i-self.num_robots]
        
        for i in range((self.num_robots + self.num_dirt1), observation.shape[0]):
            observation[i] = self.dirt_positions2[i-self.num_robots - self.num_dirt1]
        
        return observation


    def generate_dirt(self):

        # Randomly generate dirt positions in the environment
        self.num_dirt1 = np.random.randint(pow(self.grid_size,2)*0.1, pow(self.grid_size,2)*0.5)  # Randomly select the number of dirt squares
        self.num_dirt2 = np.random.randint(pow(self.grid_size,2)*0.1, pow(self.grid_size,2)*0.2)
        dirt_positions_level1 = np.empty((self.num_dirt1,2), dtype=int)
        dirt_positions_level2 = np.empty((self.num_dirt2,2), dtype=int)

        for i in range(self.num_dirt1):
            row = np.random.randint(0, self.grid_size - 1)
            col = np.random.randint(0, self.grid_size - 1)
            while np.any(np.all(np.equal(dirt_positions_level1, [row, col]), axis=1)) :
                row = np.random.randint(0, self.grid_size - 1)
                col = np.random.randint(0, self.grid_size - 1)
            dirt_positions_level1[i] = [row,col]

        for i in range(self.num_dirt2):
            row = np.random.randint(0, self.grid_size - 1)
            col = np.random.randint(0, self.grid_size - 1)

            while np.any(np.all(np.equal(dirt_positions_level1, [row, col]), axis=1)) or np.any(np.all(np.equal(dirt_positions_level2, [row, col]), axis=1)):
                row = np.random.randint(0, self.grid_size - 1)
                col = np.random.randint(0, self.grid_size - 1)
            dirt_positions_level2[i] = [row,col]
        return dirt_positions_level1, dirt_positions_level2
    

    def render(self):

        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        img = self._get_render_image()
        self.viewer.imshow(img)
        return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_render_image(self):
        scale = 20  # Scaling factor for rendering

        img = np.zeros((self.grid_size * scale, self.grid_size * scale, 3), dtype=np.uint8)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0:
                    img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale, :] = 255  # White color for empty cells
                elif self.grid[i, j] == 0.5:
                    img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale, 1] = 255  # Green color for robot positions
                elif self.grid[i, j] == 1:
                    img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale, :] = [200, 200, 200]  # Grey color for dirt squares
                elif self.grid[i, j] == 2:
                    img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale, :] = [128, 128, 128]
        return img



######################################################  MAIN  #########################################################################

# Create an instance of the cleaning environment
# env = CleaningEnv(grid_size=5, num_robots=2, max_episode_steps=100)

# # Reset the environment


# # Run the simulation


# done = False
# while not done:

#     time.sleep(0.1)
#     env.render()

#     # Change action taking into account the agent we will use ( right now is random)

#     actions = np.random.randint(env.n_actions, size=2)

#     # Take a step in the environment

#     observation, reward, _, info = env.step(actions)

#     print(f"Timestep {env.current_step}")
#     print(f"\tObservation: {observation}")
#     print(f"\tAction: {actions}\n")
    
#     if len(env.dirt_positions) == 0:
#         break


# env.close()
