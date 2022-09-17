import argparse

from click import option

parser = argparse.ArgumentParser(description="RL")


parser.add_argument(
    "--name", 
    type=str,
    default='experiment',
    help="name of the experiment to use for grouping in wandb",
)

parser.add_argument(
    "--use_wandb",
    action='store_true',
    help="whether to use wandb for logging",
)

parser.add_argument(
    "--algo",
    type=str,
    default='idqn',
    help="algorithm to run",
    choices=['idqn', 'hyperbolic_idqn', 'vdn'],
)

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
    default=100000,
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
    default=64,
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
    help="update main networks every k step",
)

# Algo parameters
parser.add_argument(
    "--lr",
    type=float,
    default=5e-4,
    help="learning rate",
)

parser.add_argument(
    "--explore_rate",
    type=float,
    default=0.75,
    help="anneals epsilon to 0.01 at (max_timesteps * explore_rate) steps",
)

parser.add_argument(
    "--buffer_size",
    type=int,
    default=10000,
    help="buffer size per agent",
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
    help="Number of times to update on batches",
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="Gamma for discounting expected rewards",
)

# Hyperbolic Discounting Parameters
parser.add_argument(
    "--number_of_gammas",
    type=int,
    default=3,
    help="Number of gammas used for hyperbolically discounting rewards",
)

parser.add_argument(
    "--gamma_max",
    type=float,
    default=0.99,
    help="Largest gamma for hyperbolic discounting",
)

parser.add_argument(
    "--hyperbolic_exponent",
    type=float,
    default=0.99,
    help="The k-coefficient for hyperbolic discounting. It is k in the equation 1 / (1 + k * t) for k > 0",
)

parser.add_argument(
    "--integral_estimate", 
    type=str,
    default='lower',
    help="The integral estimation method for calculating integral for hyperbolic discounting. Either an upper or lower rectangular Reimann sum",
    choices=['lower', 'upper'],
)

parser.add_argument(
    "--acting_policy",
    type=str,
    default='largest_gamma',
    help="The acting policy for an agent with multiple gammas used for hyperbolic discounting",
    choices=['hyperbolic', 'largest_gamma'],
)