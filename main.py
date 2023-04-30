# Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack

# Wrappers
from nes_py.wrappers import JoypadSpace

# Files
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from logger import MetricLogger

# Others
from pathlib import Path
import datetime
import torch

# Define the environment with wrappers
level = input("Choose the level (from 1-1 to 8-4): ")

if len(level) != 3 or int(level[0]) not in [1, 2, 3, 4, 5, 6, 7, 8] or level[1] != '-' or int(level[2]) not in [1, 2, 3, 4]:
    raise ValueError('Wrong level format.')

env = gym_super_mario_bros.make(f'SuperMarioBros-{level}-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

save_dir = Path('Checkpoints') / datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
save_dir.mkdir(parents=True)
checkpoint = Path('Checkpoints/2023-04-29 18-56-44/mario1.chkpt')

# Transfer Learning if agent is already trained on some levels
if input("Freeze CNN layers? (Yes, if agent is already trained on 1 level) [y/n]: ") == 'y':
    print('Loading Mario with CNN layers freezed...')
    from agent_transfer_learning import Mario
else:
    print('Loading standard Mario...')
    from agent import Mario

# Initiate Agent with Logger
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
logger = MetricLogger(save_dir)

episodes = 40000

# Training loop
for e in range(episodes):
    state = env.reset()

    while True:
        try:
            torch.cuda.empty_cache()

            action = mario.act(state) # Mario acts depending on his current state

            next_state, reward, terminated, truncated, info = env.step(action)

            mario.cache(state, next_state, action, reward, terminated) # Remember
            q, loss = mario.learn() # Learn 
            logger.log_step(reward, loss, q) # Save to logs
            state = next_state # Update the state

            if terminated or truncated or info['flag_get']:
                print(f'Reward: {logger.curr_ep_reward}')
                print(f"GPU memory usage: {torch.cuda.memory_allocated(device='cuda') / 1024**2:.2f} MB")
                break

        except KeyboardInterrupt as exc:
            save = input('Do you want to save the progress? [y/n]: ')
            if save == 'y':
                mario.save()
            env.close()
            raise SystemExit(0) from exc

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

mario.save()
env.close()