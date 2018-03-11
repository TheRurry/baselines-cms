import gym
import gym_cms

from baselines import deepq

def main():
    env = gym.make("hot-v0")
    # model = deepq.models.mlp([64], layer_norm=True)
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        # param_noise=True
    )
    print("Saving model to hc_model.pkl")
    act.save("hc_model.pkl")

if __name__ == '__main__':
    main()