import copy
import sys
import time
from itertools import count
from random import randint

import gin
import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb

from model import create_model
from utils.env_utils import create_environment, get_video_filepath
from utils.multiprocessing_utils import (
    EMPTY_MESSAGE,
    KILL_MESSAGE,
    ResultItem,
    WorkItem,
    collect_results,
    kill_processes,
    spawn_processes,
)
from utils.pytorch_utils import create_noise_tensors
from utils.utils import randint_generator, setup_logger, unzip


def record_evaluation_video(top_agent, env):
    is_recording = isinstance(env, gym.wrappers.Monitor)
    if is_recording:
        env._set_mode("evaluation")

    evaluate(top_agent, env)

    if is_recording:
        env._set_mode("training")
        env.reset()
        video_filepath = get_video_filepath(env)
        wandb.log({"Evaluate Video": wandb.Video(video_filepath)}, commit=False)


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
def mutate(model, noise_std, seed):
    new_model = copy.deepcopy(model)
    noise = create_noise_tensors(new_model, seed)
    for p, noise_p in zip(new_model.parameters(), noise):
        p.data += noise_p * noise_std
    return new_model


def mutate_and_evaluate_task(elite, work_queue, results_queue):
    env = create_environment(save_videos=False)
    while True:
        work_item = work_queue.get()
        model_id, seed, message = work_item
        if message == KILL_MESSAGE:
            return

        model = elite[model_id]
        new_model = mutate(model, seed=seed)

        fitness = evaluate(new_model, env)
        result_item = ResultItem(model_id=model_id, seed=seed, fitness=fitness)
        results_queue.put(result_item)


def spawn_mutation_work(queue, elite_size, size):
    for seed in randint_generator(size):
        model_id = randint(0, elite_size - 1)
        queue.put(WorkItem(model_id=model_id, seed=seed, message=EMPTY_MESSAGE))


def sort_population(evaluated_population):
    evaluated_population.sort(key=lambda result: result.fitness, reverse=True)


def mutate_models(elite_models, elite_ids, seeds):
    copied_models = [mutate(elite_models[elite_id], seed=seed) for elite_id, seed in zip(elite_ids, seeds)]
    for elite, mutated_elite in zip(elite_models, copied_models):
        elite.load_state_dict(mutated_elite.state_dict())
        del mutated_elite


def get_top_performers(evaluated_population, elite, elite_size):
    sort_population(evaluated_population)
    top_ids, top_seeds, top_scores = unzip(evaluated_population[:elite_size])
    mutate_models(elite, top_ids, top_seeds)
    return np.array(top_scores)


def log_generation_stats(generation, scores, time, commit=True):
    print(f"{generation}. {scores.mean():.2f} +-{scores.std():.2f}. t={time:.2f} seconds.")
    wandb.log(
        {
            "Generation": generation,
            "Elapsed Time": time,
            "Max Reward": scores.max(),
            "Mean Reward": scores.mean(),
            "Std Reward": scores.std(),
            "Min Reward": scores.min(),
        },
        commit=commit,
    )


def create_population(env, population_size):
    return [create_model(env) for _ in range(population_size)]


def get_top_performers_from_random_population(env, population, elite_size):
    evaluated_population = [(model, evaluate(model, env)) for model in population]
    evaluated_population.sort(key=lambda result: result[1], reverse=True)
    elite, top_scores = unzip(evaluated_population[:elite_size])
    for e in elite:
        e.share_memory()
    return elite, np.array(top_scores)


@gin.configurable
def train(solved_score, population_size, elite_size, num_proc, log_video_rate):
    setup_logger()
    manager = mp.Manager()
    work_queue = manager.Queue()
    results_queue = manager.Queue()

    # Random Search 1st generation
    start_time = time.time()
    env = create_environment()
    population = create_population(env, population_size)
    print(population[0])
    elite, top_scores = get_top_performers_from_random_population(env, population, elite_size)
    elapsed_time = time.time() - start_time
    log_generation_stats(1, top_scores, elapsed_time)

    # 2nd -> inf generation: Mutate Top Performers (classic GA)
    ma_reward = 0
    spawn_processes(num_proc, work_fn=mutate_and_evaluate_task, args=(elite, work_queue, results_queue))
    for generation in count(start=2, step=1):
        start_time = time.time()
        spawn_mutation_work(work_queue, elite_size, population_size)

        evaluated_population = collect_results(results_queue, size=population_size)
        top_scores = get_top_performers(evaluated_population, elite, elite_size)
        elapsed_time = time.time() - start_time
        if generation % log_video_rate == 0:
            record_evaluation_video(elite[0], env)
        log_generation_stats(generation, top_scores, elapsed_time)

        ma_reward = 0.7 * ma_reward + 0.3 * top_scores.mean()
        if ma_reward >= solved_score:
            print(f"Solved in {generation} generations")
            kill_processes(work_queue, num_proc)
            break


if __name__ == "__main__":
    experiment_file = sys.argv[1]
    gin.parse_config_file(experiment_file)
    train()
