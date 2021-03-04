import numpy as np
import torch
import random
from typing import List

class Agent:
    def act(self, observations: np.array):
        raise NotImplementedError("not implemented")

    def take_reward(self, value: float):
        raise NotImplementedError("not implemented")

    def on_end_game(self):
        raise NotImplementedError("not implemented")

class RandomAgent(Agent):

    def act(self, observations: np.array):
        return random.randint(0,1)

    def take_reward(self, value: float):
        pass

    def on_end_game(self):
        pass


class ExperiencePoint:
    def __init__(self, probabilities: torch.Tensor, reward: float):
        self.probabilities = probabilities
        self.reward = reward

class GameStory:
    def __init__(self, probabilities: torch.Tensor, rewards: torch.Tensor):
        self.probabilities: torch.Tensor = probabilities
        self.rewards: torch.Tensor = rewards


class DRLAgent(Agent):
    def __init__(self, net: torch.nn.Module, discount=0.99, learning_rate=0.01):
        self.net = net
        self.experience : List[ExperiencePoint] = []
        self.games : List[GameStory] = []
        self.discount = discount

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

    def act(self, observations: np.array):
        prediction = self.net(torch.tensor(observations))
        action = np.random.choice(range(len(prediction)), p=prediction.detach())
        self.experience.append(ExperiencePoint(prediction[action], 0.0))
        return action

    def take_reward(self, value: float):
        self.experience[-1].reward = value

    def on_end_game(self):
        self.discount_future_rewards()
        predicted = torch.cat([e.probabilities.reshape(1) for e in self.experience]).double()
        rewards = torch.tensor([e.reward for e in self.experience])
        self.games.append(GameStory(
            probabilities=predicted,
            rewards=rewards
        ))
        self.experience = []


    def discount_future_rewards(self):
        # r0, r1, r2, ...
        # rd0 = sum(ri * g^i) = r0 + g * r1 + g^2 * r2 + ...
        # rdn = rn
        crt_discounted = 0
        for exp in reversed(self.experience):
            crt_discounted = exp.reward + self.discount * crt_discounted
            exp.reward = crt_discounted

    def optimize(self):
        rewards   = torch.cat([story.rewards       for story in self.games])
        predicted = torch.cat([story.probabilities for story in self.games])

        loss = -(predicted.log() * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.games = []
