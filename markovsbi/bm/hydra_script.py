import hydra
import logging
from markovsbi.bm.api_utils import set_diffusion_method, set_factorization_method

from omegaconf import DictConfig, OmegaConf

import os
import sys
import socket
import time

# Backends
import jax
import torch
import numpy as np
import random


from markovsbi.tasks import get_task
from markovsbi.bm.data_utils import (
    generate_unique_model_id,
    init_dir,
    load_model,
    save_model,
    save_summary,
)
from markovsbi.bm.eval import eval

logo = """
░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒▒▓█▓▒░  
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓███████▓▒░░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒▒▓█▓▒░  
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▓█▓▒░   
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ ░▒▓█▓▓█▓▒░   
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░   ░▒▓██▓▒░    
"""


def main():
    """Main script function"""
    print(logo)
    _main()


@hydra.main(config_path="../config", config_name="config.yaml", version_base=None)
def _main(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Go back to the folder named "cfg.name"
    output_super_dir = os.path.dirname(output_dir)
    while os.path.basename(output_super_dir) != cfg.name:
        output_super_dir = os.path.dirname(output_super_dir)

    init_dir(output_super_dir)

    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info("Output super directory: {}".format(output_super_dir))
    log.info(f"Hostname: {socket.gethostname()}")
    try:
        log.info(f"Jax devices: {jax.devices()}")
    except:
        # Set devices to CPU
        jax.config.update("jax_platform_name", "cpu")
        log.info(f"Jax devices: {jax.devices()}")
    log.info(f"Torch devices: {torch.cuda.device_count()}")

    seed = cfg.seed
    rng = set_seed(seed)
    backend = cfg.method.backend

    # If model id is specified, no training!

    # Initialize task
    log.info(f"Task: {cfg.task.name}")
    rng, rng_task = jax.random.split(rng)
    task_name = cfg.task.name
    task = get_task(task_name, **dict(cfg.task.params))
    num_simulations = cfg.task.num_simulations
    num_steps = cfg.task.num_steps
    data = task.get_data(rng_task, cfg.task.num_simulations, cfg.task.num_steps)
    # Convert data to backend
    data = {k: convert_to_backend(v, backend) for k, v in data.items()}

    # Load model if model_id is specified
    if cfg.model_id is not None:
        model_id = cfg.model_id
        log.info(f"Loading model {model_id}")
        model = load_model(output_super_dir, model_id)
        training_time = None

        new_cfg = model.cfg
        # Update with current evaluation config
        new_cfg.eval = cfg.eval
        new_cfg.save_summary = cfg.save_summary
        new_cfg.save_model = cfg.save_model
        new_cfg.model_id = cfg.model_id
        new_cfg.method.sampler = cfg.method.sampler

        cfg = new_cfg

    # Run method
    log.info(f"Method: {cfg.method.name}")
    rng, rng_method = jax.random.split(rng)
    method = cfg.method.name

    if method == "nle" or method == "nre":
        from markovsbi.bm.sbi_baselines import run_factorized_nle_or_nre

        if cfg.model_id is None:
            start_time = time.time()
            model = run_factorized_nle_or_nre(cfg, task, data, method)
            end_time = time.time()
            training_time = end_time - start_time

        # Set sampler
        sampler_name = cfg.method.sampler.name

    elif method == "npe":
        from markovsbi.bm.sbi_baselines import run_npe_embedding_network

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_npe_embedding_network(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time}")

        # Set sampler
        sampler_name = cfg.method.sampler.name

    elif method == "nse":
        from markovsbi.bm.sbi_baselines import run_nse_embedding_network

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_nse_embedding_network(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time}")

        # Set sampler
        sampler_name = cfg.method.sampler.name

    elif method == "npe_summary":
        from markovsbi.bm.sbi_baselines import run_npe_sufficient_summary_stat

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_npe_sufficient_summary_stat(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time}")

        # Set sampler
        sampler_name = cfg.method.sampler.name
    elif method == "nle_summary":
        from markovsbi.bm.sbi_baselines import run_nle_sufficient_summary_stat

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_nle_sufficient_summary_stat(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time}")

        # Set sampler
        sampler_name = cfg.method.sampler.name
    elif method == "npe_sliced_summary":
        from markovsbi.bm.sbi_baselines import run_npe_sliced_sufficient_summary_stat

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_npe_sliced_sufficient_summary_stat(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time}")

        # Set sampler
        sampler_name = cfg.method.sampler.name
    elif method == "nle_sliced_summary":
        from markovsbi.bm.sbi_baselines import run_nle_sliced_sufficient_summary_stat

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_nle_sliced_sufficient_summary_stat(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training time: {training_time}")

        # Set sampler
        sampler_name = cfg.method.sampler.name

    elif method == "diffusion":
        from markovsbi.bm.diffusion_baselines import run_train_factorized_diffusion

        # Train or load model
        if cfg.model_id is None:
            start_time = time.time()
            model = run_train_factorized_diffusion(cfg, task, data, rng_method)
            end_time = time.time()
            training_time = end_time - start_time

        # Set sampler
        sampler_name = cfg.method.sampler.name
        set_factorization_method(cfg, model)
        set_diffusion_method(cfg, model)
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    # Update task if necessary
    task_name = cfg.task.name
    task = get_task(task_name)

    # Evaluate model
    log.info("Evaluating model")
    rng, rng_eval = jax.random.split(rng)
    eval_methods = cfg.eval.name

    if eval_methods[0] is None:
        num_steps_xs = cfg.eval.num_steps_xs
        metric_values = [[None] * len(num_steps_xs)] * len(eval_methods)
        sampling_times = [[None] * len(num_steps_xs)] * len(eval_methods)
    else:
        num_steps_xs = cfg.eval.num_steps_xs
        metric_values, sampling_times = eval(cfg, eval_methods, model, task, rng_eval)
        for eval_method, metric_value in zip(eval_methods, metric_values):
            log.info(f"Metric {eval_method}: {metric_value}")

    # Saving results
    is_save_model = cfg.save_model
    is_save_summary = cfg.save_summary

    if is_save_model and (cfg.model_id is None):
        model_id = generate_unique_model_id(output_super_dir)
        log.info(f"Saving model with id {model_id}")
        try:
            save_model(model, output_super_dir, model_id)
        except Exception as e:
            log.error(f"Error saving model: {e}")
            # Print traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            import traceback

            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout
            )

    if is_save_summary:
        for i, eval_method in enumerate(eval_methods):
            metric_value = metric_values[i]
            sampling_time = sampling_times[i]

            for s, m in zip(num_steps_xs, metric_value):
                log.info("Saving summary")
                save_summary(
                    output_super_dir,
                    method,
                    sampler_name,
                    task_name,
                    num_simulations,
                    num_steps,
                    model_id,
                    eval_method,
                    m,
                    s,
                    seed,
                    training_time,
                    sampling_time,
                    cfg,
                )

    return sum([m if m is not None else 0 for m in metric_value]) / len(metric_value)


def set_seed(seed: int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(seed)
    return key


def convert_to_backend(x: jax.Array | torch.Tensor, backend: str):
    """Converts the input to the backend."""
    if backend == "torch":
        x = torch.tensor(np.array(x))
    elif backend == "jax":
        x = jax.numpy.array(np.array(x))
    return x
