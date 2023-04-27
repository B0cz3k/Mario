# Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import FrameStack

# Wrappers
from nes_py.wrappers import JoypadSpace

# Files
from agent import Mario
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from logger import MetricLogger

# Others
from pathlib import Path
import datetime

# Define the environment with wrappers
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = SkipFrame(env, skip=5)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=5)

save_dir = Path('Checkpoints') / datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
save_dir.mkdir(parents=True)
checkpoint = None # Path('Checkpoints/2023-04-28 00-50-31/mario0.chkpt')

# Initiate Agent with Logger
mario = Mario(state_dim=(5, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
logger = MetricLogger(save_dir)

episodes = 40000

# Training loop
for e in range(episodes):
    state = env.reset()

    while True:
        try:
            action = mario.act(state) # Mario acts depending on his current state

            next_state, reward, terminated, truncated, info = env.step(action)

            mario.cache(state, next_state, action, reward, terminated) # Remember
            q, loss = mario.learn() # Learn 
            logger.log_step(reward, loss, q) # Save to logs
            state = next_state # Update the state

            if terminated or truncated or info['flag_get']:
                print(f'Reward: {reward}')
                break

        except KeyboardInterrupt as exc:
            save = input('Do you want to save the progress? [y/n]')
            if save == 'y':
                mario.save()
            env.close()
            raise SystemExit(0) from exc

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

mario.save()
env.close()