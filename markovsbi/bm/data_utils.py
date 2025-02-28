import torch
import jax.numpy as jnp

import numpy as np
import math

import os
import pandas as pd

import pickle

from filelock import FileLock


def init_dir(dir_path: str):
    """Initializes a directory for storing models and summary.csv"""
    if not os.path.exists(dir_path + os.sep + "models"):
        os.makedirs(dir_path + os.sep + "models")

    if not os.path.exists(dir_path + os.sep + "summary.csv"):
        df = pd.DataFrame(
            columns=[
                "method",
                "sampler",
                "task",
                "num_simulations",
                "num_steps",
                "seed",
                "model_id",
                "metric",
                "value",
                "eval_num_steps",
                "time_train",
                "time_eval",
                "cfg",
            ]
        )
        df.to_csv(dir_path + os.sep + "summary.csv", index=False)


def get_summary_df(dir_path):
    """Returns the summary.csv file as a pandas dataframe."""
    df = pd.read_csv(dir_path + os.sep + "summary.csv")
    return df


def generate_unique_model_id(dir_path):
    """Generates a unique model id for saving a model."""
    summary_df = get_summary_df(dir_path)
    model_ids = summary_df["model_id"].values
    if len(model_ids) == 0:
        return 0
    elif len(model_ids) == 1:
        return 1
    else:
        max_id = np.max(model_ids)
        return max_id + 1


def save_model(model, dir_path, model_id):
    """Saves a model to a file."""
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, "wb") as file:
        pickle.dump(model, file)


def save_summary(
    dir_path,
    method: str,
    sampler: str,
    task: str,
    num_simulations: int,
    num_steps: int,
    model_id: int,
    metric: str,
    value: float,
    eval_num_steps: int,
    seed: int,
    time_train: float,
    time_eval: float,
    cfg: dict,
):
    """Saves a summary to the summary.csv file with thread-safe file locking."""
    summary_file = os.path.join(dir_path, "summary.csv")
    lock_file = summary_file + ".lock"  # Creates a lock file next to the CSV

    # Create a file lock object
    lock = FileLock(lock_file)

    # Acquire the lock before writing
    with lock:
        summary_df = get_summary_df(dir_path)
        new_row = pd.DataFrame(
            {
                "method": method,
                "sampler": sampler,
                "task": task,
                "num_simulations": num_simulations,
                "num_steps": num_steps,
                "seed": seed,
                "model_id": model_id,
                "metric": metric,
                "value": str(value),
                "eval_num_steps": eval_num_steps,
                "time_train": str(time_train),
                "time_eval": str(time_eval),
                "cfg": str(cfg),
            },
            index=[len(summary_df)],
        )
        summary_df = pd.concat([summary_df, new_row], axis=0, ignore_index=True)
        summary_df.to_csv(summary_file, index=False)


def load_model(dir_path, model_id):
    """Loads a model from a file."""
    file_name = dir_path + os.sep + "models" + os.sep + f"model_{model_id}.pkl"
    with open(file_name, "rb") as file:
        return pickle.load(file)


def query(
    name,
    method=None,
    sampler=None,
    task=None,
    num_simulations=None,
    num_steps=None,
    seed=None,
    metric=None,
    eval_num_steps=None,
    **kwargs,
):
    """Queries the summary.csv file."""
    summary_df = get_summary_df(name)
    query = ""
    query += to_query_string("method", method)
    query += to_query_string("sampler", sampler)
    query += to_query_string("task", task)
    query += to_query_string("num_simulations", num_simulations)
    query += to_query_string("num_steps", num_steps)
    query += to_query_string("seed", seed)
    query += to_query_string("metric", metric)
    query += to_query_string("eval_num_steps", eval_num_steps)
    if query.endswith(" & "):
        query = query[:-3]
    if query == "":
        return summary_df
    else:
        print(query)
        return summary_df.query(query)


def to_query_string(name: str, var, end=" & ") -> str:
    """Translates a variable to string.

    Args:
        name (str): Query argument
        var (str): value

    Returns:
        str: Query == value ?
    """
    if var is None:
        return ""
    elif (
        var is pd.NA
        or var is torch.nan
        or var is math.nan
        or str(var) == "nan"
        or var is jnp.nan
    ):
        return f"{name}!={name}"
    elif isinstance(var, list) or isinstance(var, tuple):
        query = "("
        for v in var:
            if query != "(":
                query += "|"
            if isinstance(v, str):
                query += f"{name}=='{v}'"
            else:
                query += f"{name}=={v}"
        query += ")"
    else:
        if isinstance(var, str):
            query = f"({name}=='{var}')"
        else:
            query = f"({name}=={var})"
    return query + end
