idqn_params = {'env_name': 'ma_gym:PongDuel-v0',
            'lr': 0.0005,
            'batch_size': 32,
            'gamma': 0.99,
            'buffer_limit': 50000,
            'log_interval': 100,
            'max_episodes': 500,
            'max_epsilon': 0.9,
            'min_epsilon': 0.1,
            'test_episodes': 10,
            'warm_up_steps': 2000,
            'update_iter': 10,
            'monitor': True}


qmix_params = {'env_name': None,
              'lr': 0.001,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'update_target_interval': 20,
              'log_interval': 100,
              'max_episodes': 500,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
              'recurrent': False}

vdn_params = {'env_name': None,
              'lr': 0.001,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'update_target_interval': 20,
              'log_interval': 100,
              'max_episodes': 500,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
              'recurrent': False}

# TODO: Add maddpg (currently in the original repo, says maddpg is having issues)