import copy
import logging
import random
import sys
import time

import gin
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from model import Model
from utils.env_utils import create_environment
from utils.utils import unzip

NOISE_STD = 0.01
POPULATION_SIZE = 50
PARENTS_COUNT = 10
NUM_PROC = 4


def evaluate(model, env):
    total_reward = 0
    o = env.reset()
    done = False
    while not done:
        o_tensor = torch.FloatTensor(o).unsqueeze(0)
        a = model(o_tensor)
        a = a.item()
        o, reward, done, _ = env.step(a)
        total_reward += reward

    return total_reward


def mutate(model):
    noise = create_noise_tensor(model)
    new_model = copy.deepcopy(model)
    for p, noise_p in zip(new_model.parameters(), noise):
        p.data += noise_p * NOISE_STD
    return new_model


def create_noise_tensor(model):
    return [torch.normal(mean=0, std=1, size=p.data.size()) for p in model.parameters()]


def mutate_new_models(top_models):
    return [mutate(random.choice(top_models)) for _ in range(POPULATION_SIZE)]


@gin.configurable
def main(solved_score):
    env = create_environment()
    population = [Model() for _ in range(POPULATION_SIZE)]
    print(population[0])

    ma_reward = 0
    for epoch in range(1000):
        start_time = time.time()
        evaluated_population = [(model, evaluate(model, env)) for model in population]
        evaluated_population.sort(key=lambda model: model[1], reverse=True)
        top_models, top_scores = unzip(evaluated_population[:PARENTS_COUNT])

        top_scores = np.array(top_scores)
        # wandb.log({"Max Reward": top_scores.max(), "Mean Reward": top_scores.mean(), "Std Reward": top_scores.std()})
        ma_reward = 0.5 * ma_reward + 0.5 * top_scores.mean()
        if ma_reward >= solved_score:
            print(f"Solved in {epoch} epochs")
            break

        population = mutate_new_models(top_models)
        epoch_time = time.time() - start_time
        # wandb.log(
        #     {"Epoch": epoch, "Rolling Mean Reward": ma_reward, "Epoch Time": epoch_tiem}
        # )
        print(f"{epoch}. {top_scores.mean():.2f} +-{top_scores.std():.2f}. t={epoch_time:.2f} seconds.")


if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    main()
