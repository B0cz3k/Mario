'''You can go through possible levels by changing the env variable, e.g. SuperMarioBros-8-1'''

# Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Wrappers
from nes_py.wrappers import JoypadSpace

# Define the environment with wrappers
env = gym_super_mario_bros.make('SuperMarioBros-4-1-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

while True:
    next_state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
    if terminated or truncated or info['flag_get']:
        print(f'Reward: {reward}')
        break
        
env.close()