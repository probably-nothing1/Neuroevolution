import copy
import random
import sys
import time
from collections import namedtuple
from itertools import count
from random import randint

import gin
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from model import create_model
from utils.env_utils import create_environment
from utils.utils import set_seed, unzip

EMPTY_MESSAGE = b""
KILL_MESSAGE = b"KILL"
WorkItem = namedtuple("WorkItem", field_names=["seeds", "message"])
ResultItem = namedtuple("ResultItem", field_names=["seeds", "fitness"])


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
def mutate(model, noise_std, seed=None):
    noise = create_noise_tensor(model, seed)
    new_model = copy.deepcopy(model)
    for p, noise_p in zip(new_model.parameters(), noise):
        p.data += noise_p * noise_std
    return new_model


def create_noise_tensor(model, seed=None):
    if seed:
        set_seed(seed)
    return [torch.normal(mean=0, std=1, size=p.data.size()) for p in model.parameters()]


def recreate_model(seeds, env):
    if not seeds:
        raise ValueError(f"Seeds list: {seeds}")
    set_seed(seeds[0])
    model = create_model(env)
    for seed in seeds[1:]:
        model = mutate(model, seed=seed)
    return model


def worker_task(work_queue, results_queue):
    env = create_environment()
    while True:
        work_item = work_queue.get()
        if work_item.message == KILL_MESSAGE:
            return
        model = recreate_model(work_item.seeds, env)
        fitness = evaluate(model, env)
        result_item = ResultItem(seeds=work_item.seeds, fitness=fitness)
        results_queue.put(result_item)


## -------------------
@gin.configurable
def create_population(top_models=None, population_size=50):
    seeds = generate_list_of_seeds_sequence(population_size)
    if top_models is None:
        return seeds
    return [random.choice(top_models) + seed for seed in seeds]


def generate_list_of_seeds_sequence(size):
    return [[randint(0, 2 ** 32 - 1)] for _ in range(size)]


def spawn_processes(num_proc, work_fn, args):
    for _ in range(num_proc):
        mp.Process(target=work_fn, args=args).start()


def spawn_work(population, work_queue):
    for model in population:
        work_queue.put(WorkItem(model, EMPTY_MESSAGE))


def sort_population(evaluated_population):
    evaluated_population.sort(key=lambda model: model.fitness, reverse=True)


@gin.configurable
def get_top_performers(evaluated_population, top_count):
    sort_population(evaluated_population)
    top_models, top_scores = unzip(evaluated_population[:top_count])
    return top_models, np.array(top_scores)


def collect_results(results_queue, population_size):
    return [results_queue.get() for _ in range(population_size)]


def kill_processes(work_queue, num_proc):
    [work_queue.put(WorkItem([], KILL_MESSAGE)) for _ in range(num_proc)]


@gin.configurable
def train(solved_score, population_size, num_proc):
    manager = mp.Manager()
    work_queue = manager.Queue()
    results_queue = manager.Queue()
    spawn_processes(num_proc, work_fn=worker_task, args=(work_queue, results_queue))

    population = create_population(population_size=population_size)
    ma_reward = 0
    for generation in count(start=1, step=1):
        start_time = time.time()
        spawn_work(population, work_queue)
        evaluated_population = collect_results(results_queue, population_size)
        top_models, top_scores = get_top_performers(evaluated_population)

        population = create_population(top_models, population_size=population_size)
        epoch_time = time.time() - start_time
        # wandb.log({"Max Reward": top_scores.max(), "Mean Reward": top_scores.mean(), "Std Reward": top_scores.std()})
        ma_reward = 0.5 * ma_reward + 0.5 * top_scores.mean()
        print(f"{generation}. {top_scores.mean():.2f} +-{top_scores.std():.2f}. t={epoch_time:.2f} seconds.")
        # wandb.log(
        #     {"Epoch": epoch, "Rolling Mean Reward": ma_reward, "Epoch Time": epoch_tiem}
        # )
        if ma_reward >= solved_score:
            print(f"Solved in {generation} generations")
            kill_processes(work_queue, num_proc)
            break


if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    train()
