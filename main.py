#!/usr/bin/env python3
import gym
import torch
from agent import Agent, DRLAgent, RandomAgent

from anim_plot import GraphAnimation

class World:

    def __init__(self, env: gym.core.Env, agent: Agent):
        self.env = env
        self.agent = agent
        self.reset_env()

    def step(self):
        self.last_observation, self.last_reward, self.done, self.last_info = self.env.step(
            self.agent.act(self.last_observation)
        )
        self.agent.take_reward(self.last_reward)
        if self.done:
            self.agent.on_end_game()


    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def reset_env(self):
        self.done = False
        self.last_reward = 0.0
        self.last_observation = self.env.reset()
        self.last_info = None

class LambdaLayer(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, batch):
        return self.func(batch)


if __name__ == '__main__':
    # env : gym.core.Env = gym.make('CartPole-v1')
    # net = torch.nn.Sequential(
    #     LambdaLayer(lambda batch: batch.float()),
    #     torch.nn.Linear(env.observation_space.shape[0], 8),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(8, 8),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(8, env.action_space.n),
    #     torch.nn.Softmax(dim=-1)
    # )
    env : gym.core.Env = gym.make('Pong-v0')
    net = torch.nn.Sequential(
        LambdaLayer(lambda batch: batch.T[None, ...].float()),
        torch.nn.Conv2d(3,1,3),
        torch.nn.Conv2d(1,1,3),
        torch.nn.MaxPool2d(8),
        torch.nn.Flatten(),
        torch.nn.Linear(475, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, env.action_space.n),
        torch.nn.Softmax(dim=-1)
    )
    drl_agent = DRLAgent(net=net)

    world = World(env, drl_agent)
    # anim = GraphAnimation('avg total reward', '')

    for i in range(300):
        total_reward = 0
        for j in range(1):
            world.reset_env()
            while not world.done:
                world.render()
                total_reward += world.last_reward
                world.step()

        # anim.extend_line1([i], [total_reward/10])
        # if i % 10 == 0:
        #     anim.redraw()
        drl_agent.optimize()