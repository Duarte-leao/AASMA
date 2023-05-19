

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
        self.dirt_positions = []
        self.viewer = None

        self.current_step = 0


    def reset(self):

        # Reset the environment
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.robot_positions = np.zeros((self.num_robots, 2), dtype=int)
        self.dirt_positions = self.generate_dirt()

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
                if np.any(np.all(self.dirt_positions == robot_pos, axis=0)):
                    print(self.dirt_positions)
                    self.dirt_positions = self.dirt_positions[self.dirt_positions != robot_pos]
                    print('REMOVEU DIRT:',self.dirt_positions, 'ROBOT: ', robot_pos )   
                    # mask = np.any(np.all(dirt_positions == rob_pos, axis=1), axis=0)
                    # dirt_positions = dirt_positions[~mask]

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

        for i in range(self.num_robots):
            print('robot position',self.robot_positions[i] )
            row, col = self.robot_positions[i]
            self.grid[row, col] = 1

        for i in range(self.num_dirt):
            print(self.dirt_positions[i])
            # row, col = self.dirt_positions[i]
            self.grid[self.dirt_positions[i,0], self.dirt_positions[i,1]] = 0.5  # Set dirt squares to grey
            # print(self.grid)

        return 
    

    # Should chage it ?? Create a better reward function ??? (not for right now)
    def _calculate_reward(self):
  
        cleaned_area = np.sum(self.grid == 0.5)
        total_area = self.grid_size * self.grid_size
        return 1 - (cleaned_area / total_area)
    

    def get_observation(self):

        # Get the current observation. THIS OBSRVATION HAS THE ROBOT POSITION AND DIRT POSITION [[agent1_pos], [agent2_pos], [dirt1],... ,[dirt n]]

        observation = np.zeros((self.num_robots + self.num_dirt, 2 ), dtype=int)  
        for i in range(self.num_robots):
            observation[i] = self.robot_positions[i]

        for i in range(self.num_robots, (self.num_robots + self.num_dirt)):
            observation[i] = self.dirt_positions[i-self.num_robots]

        return observation


    # def generate_dirt(self):
    #     # Randomly generate dirt positions in the environment
    #     num_dirt = random.randint(1, self.grid_size)  # Randomly select the number of dirt squares
    #     dirt_positions = []

    #     for i in range(num_dirt):
    #         row = random.randint(0, self.grid_size - 1)
    #         col = random.randint(0, self.grid_size - 1)
    #         dirt_positions.append([row, col])

    #     return dirt_positions

    def generate_dirt(self):

        # Randomly generate dirt positions in the environment
        self.num_dirt = np.random.randint(1, self.grid_size)  # Randomly select the number of dirt squares
        dirt_positions = np.empty((self.num_dirt,2), dtype=int)
        print(self.num_dirt)
        for i in range(self.num_dirt):
            row = np.random.randint(0, self.grid_size - 1)
            col = np.random.randint(0, self.grid_size - 1)
            dirt_positions[i] = [row,col]
        print('real dirt positions:', dirt_positions)
        return dirt_positions
    

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
                elif self.grid[i, j] == 1:
                    img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale, 1] = 255  # Green color for robot positions
                elif self.grid[i, j] == 0.5:
                    img[i * scale: (i + 1) * scale, j * scale: (j + 1) * scale, :] = [128, 128, 128]  # Grey color for dirt squares

        return img


######################################################  MAIN  #########################################################################

# Create an instance of the cleaning environment
env = CleaningEnv(grid_size=10, num_robots=2, max_episode_steps=100)

# Reset the environment
observation = env.reset()

# Run the simulation
done = False
while not done:

    time.sleep(0.8)
    env.render()

    # Change action taking into account the agent we will use ( right now is random)

    actions = np.random.randint(env.n_actions, size=2)

    # Take a step in the environment

    observation, reward, _, info = env.step(actions)

    print(f"Timestep {env.current_step}")
    # print(f"\tObservation: {observation}")
    print(f"\tAction: {actions}\n")

env.close()
