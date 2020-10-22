from collections import namedtuple

import torch.multiprocessing as mp

EMPTY_MESSAGE = b""
KILL_MESSAGE = b"KILL"
WorkItem = namedtuple("WorkItem", field_names=["model_id", "seed", "message"])
ResultItem = namedtuple("ResultItem", field_names=["model_id", "seed", "fitness"])


def spawn_processes(num_proc, work_fn, args):
    processes = [mp.Process(target=work_fn, args=args) for _ in range(num_proc)]
    for p in processes:
        p.start()
    return processes


def kill_processes(queue, num_proc):
    for _ in range(num_proc):
        queue.put(WorkItem(None, None, KILL_MESSAGE))


def collect_results(queue, size):
    return [queue.get() for _ in range(size)]
