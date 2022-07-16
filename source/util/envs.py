from pettingzoo.butterfly import knights_archers_zombies_v10, cooperative_pong_v5
from pettingzoo.mpe import simple_v2
from pettingzoo.utils.conversions import aec_to_parallel
from supersuit import (
    black_death_v3,
    concat_vec_envs_v1,
    flatten_v0,
    pettingzoo_env_to_vec_env_v1,
    color_reduction_v0,
)

def get_env(env_name):
    image_input = False # Determines if CNN are used or not
    if env_name == "kaz":
        env = flatten_v0(black_death_v3(knights_archers_zombies_v10.env(use_typemasks=True)))
    elif env_name == "simple":
        env = flatten_v0(simple_v2.env(max_cycles=50, continuous_actions=False))
    elif env_name == "pong":
        env = color_reduction_v0(cooperative_pong_v5.env(), mode="full")
        image_input = True
    else:
        raise ValueError("Unknown environment name: {}".format(env_name))
    env.reset()
    agent_names = env.agents
    return env, agent_names, image_input


def parallel_env(env, num_envs):
    env = aec_to_parallel(env)
    env = pettingzoo_env_to_vec_env_v1(env)
    env = concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="gym")
    return env