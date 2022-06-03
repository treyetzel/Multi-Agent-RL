from algorithms.idqn import run_idqn
from util.default_params import idqn_params, qmix_params, vdn_params

# ===== Set up experiment variables =======
# TODO: add arguments parse so we can get values from command line
USE_WANDB = False #TODO explore WANDB
algorithm_type = "idqn"
env = "ma_gym:PongDuel-v0"
# =========================================


if algorithm_type == "idqn":
    kwargs = idqn_params
    run = run_idqn
elif algorithm_type == "qmix":
    kwargs = qmix_params
    run = None # Just working on IDQN for now
elif algorithm_type == "vdn":
    kwargs = vdn_params
    run = None


# common hyperparams:
kwargs["env_name"] = env
kwargs["monitor"] = False
kwargs["max_episodes"] = 500
kwargs["log_interval"] = 100
kwargs["test_episodes"] = 10
kwargs["batch_size"] = 32
kwargs["lr"] = 0.001
kwargs["buffer_limit"] = 50000
kwargs["USE_WANDB"] = USE_WANDB

print(kwargs)
if USE_WANDB:
    import wandb
    wandb.init(project='minimal-marl', config={'algo': algorithm_type, **kwargs}, monitor_gym=True)

run(**kwargs)
