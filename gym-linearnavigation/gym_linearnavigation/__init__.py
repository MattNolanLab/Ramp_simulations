from gym.envs.registration import register


register(
    id='LinearNavigationVisualOriginal-v2',
    entry_point='gym_linearnavigation.envs:LinearNavigationVisualV2',
    kwargs={'is_probe': False, 'num_middle_repetitions': 10, 'n_trials': 10, 'max_reward_displacement': 0, 'rz_start': 430, 'p_image_noise': 0.075}
)
