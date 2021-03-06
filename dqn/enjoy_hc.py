import gym
import gym_cms

from baselines import deepq

def main():
    env = gym.make("hot-v0")
    act = deepq.load("hc_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()