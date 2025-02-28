from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "numpy >= 1.11.1",
    "matplotlib >= 1.5.1",
    "jax[cpu] == 0.4.29",
    "optax == 0.2.2",
    "dm-haiku",
    "jupyter",
    "jax_cfd",
    "pillow",
    "seaborn",
    "jaxtyping",
]

ENTRY_POINTS = {
    "console_scripts": [
        "markovsbi = markovsbi.bm:main",
    ],
}

EXTRAS = {
    "cuda": [
        "jax[cuda12] == 0.4.29",
        "jaxlib[cuda12]",
    ],
    "bm": [
        "jax[cuda12] == 0.4.29",
        "jaxlib[cuda12]",
        "torch@https://download.pytorch.org/whl/cpu/torch-2.4.0%2Bcpu-cp311-cp311-linux_x86_64.whl",  # This will only work on linux
        "blackjax==1.2.1",
        "sbi == 0.23.1",
        "hydra-core",
        "hydra-submitit-launcher",
        "hydra-optuna-sweeper",
    ],
    "dev": [
        "autoflake",
        "black",
        "flake8",
        "isort>5.0.0",
        "ipdb",
        "pytest",
        "pytest-plt",
        "typeguard",
    ],
}


setup(
    name="markovsbi",
    version="0.0.1",
    author="na",
    description="marcovsbi is a package for sbi for Markov chains.",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS,
    entry_points=ENTRY_POINTS,
)
