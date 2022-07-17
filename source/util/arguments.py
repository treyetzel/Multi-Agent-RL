import argparse

from click import option

parser = argparse.ArgumentParser(description="RL")

# Training loop parameters
parser.add_argument(
    "--env", 
    type=str,
    default='kaz',
    help="name of environment",
    choices=['kaz', 'simple', 'pong'],
)

parser.add_argument(
    "--seed", 
    type=int,
    default=0,
    help="sets random seed for all random number generators and environments to get reproducible results",
)

parser.add_argument(
    "--max_steps", 
    type=int,
    default=1000000,
    help="max steps to train models on"
)

parser.add_argument(
    "--log_steps",
    type=int,
    default=2000,
    help="Number of steps to log"
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=128,
    help="Number of parallel environments to collect experiences from",
)

parser.add_argument(
    "--warm_up_steps",
    type=int,
    default=2000,
    help="Number of steps to explore before training",
)

parser.add_argument(
    "--k_step",
    type=int,
    default=4,
    help="update main networks every k step"
)

# IDQN parameters
parser.add_argument(
    "--lr",
    type=float,
    default=5e-4,
    help="learning rate")

parser.add_argument(
    "--buffer_size",
    type=int,
    default=10000,
    help="buffer size per agent"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="batch size to sample from replay buffers",
)

parser.add_argument(
    "--num_updates",
    type=int,
    default=10,
    help="Number of times to update on batches"
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Gamma for discounting expected rewards"
)
