import copy
import random
import sys
import time

import gin
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from model import create_model
from utils.env_utils import create_environment
from utils.utils import unzip


def evaluate(model, env):
    total_reward = 0
    o = env.reset()
    done = False
    while not done:
        o_tensor = torch.FloatTensor(o).unsqueeze(0)
        a = model(o_tensor)
        o, reward, done, _ = env.step(a)
        total_reward += reward

    return total_reward


@gin.configurable
def mutate(model, noise_std):
    noise = create_noise_tensor(model)
    new_model = copy.deepcopy(model)
    for p, noise_p in zip(new_model.parameters(), noise):
        p.data += noise_p * noise_std
    return new_model


def create_noise_tensor(model):
    return [torch.normal(mean=0, std=1, size=p.data.size()) for p in model.parameters()]


@gin.configurable
def create_population(env, population_size=50, top_models=None):
    if top_models:
        return [mutate(random.choice(top_models)) for _ in range(population_size)]
    return [create_model(env) for _ in range(population_size)]


def evaluate_population(population, env):
    return [(model, evaluate(model, env)) for model in population]


def sort_population(evaluated_population):
    evaluated_population.sort(key=lambda model: model[1], reverse=True)


@gin.configurable
def get_top_performers(evaluated_population, top_count):
    sort_population(evaluated_population)
    top_models, top_scores = unzip(evaluated_population[:top_count])
    return top_models, np.array(top_scores)


@gin.configurable
def train(solved_score, num_proc):
    env = create_environment()
    population = create_population(env)

    ma_reward = 0
    for epoch in range(1000):
        start_time = time.time()
        evaluated_population = evaluate_population(population, env)
        top_models, top_scores = get_top_performers(evaluated_population)

        # wandb.log({"Max Reward": top_scores.max(), "Mean Reward": top_scores.mean(), "Std Reward": top_scores.std()})
        ma_reward = 0.5 * ma_reward + 0.5 * top_scores.mean()
        if ma_reward >= solved_score:
            print(f"Solved in {epoch} epochs")
            break

        population = create_population(env, top_models=top_models)
        epoch_time = time.time() - start_time
        # wandb.log(
        #     {"Epoch": epoch, "Rolling Mean Reward": ma_reward, "Epoch Time": epoch_tiem}
        # )
        print(f"{epoch}. {top_scores.mean():.2f} +-{top_scores.std():.2f}. t={epoch_time:.2f} seconds.")


if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    train()
