import itertools
import random
import numpy as np
import sys
import scipy.stats
import json
import time
import subprocess
from pathlib import Path

# parameters
EXPERIMENT_NAME = 'linearnavigation_visual_paper'

seed = np.random.randint(1, 1000000)
dropout = 0
recurrent_dropout = 0
weight_decay = 5e-07
n_neurons = 512
l1 = 0
l1_rate = 0
l2_rate = 0

num_env_steps = 50_000_000


if (len(sys.argv) > 1) and ('--nocuda' in sys.argv[1]):
    cuda = False
    print(f'Cuda disabled for this run with --nocuda flag')
else:
    cuda = True


_ = '__-__'  # Delimiter

model_path = Path(f'results/{EXPERIMENT_NAME}/models/SEED={seed}{_}DROPOUT={dropout}{_}WEIGHT_DECAY={weight_decay}{_}N_NEURONS={n_neurons}{_}L1={l1}{_}RECURRENT_DROPOUT={recurrent_dropout}{_}L1_RATE={l1_rate}{_}L2_RATE={l2_rate}/')
model_path.mkdir(parents=True, exist_ok=True)

if (model_path / 'training.done').exists():
    print(f'Skipping - training done for {model_path}')
else:
    save_dir = model_path / 'weights'
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_checkpoints = list(save_dir.glob('*.pt'))
    if len(existing_checkpoints) > 0:
        print(f'Deleting existing checkpoints')
        print(existing_checkpoints)
        [p.unlink() for p in existing_checkpoints]

    command = f'''python3 -u models/pytorch-a2c-ppo-acktr-gail/main.py 
            --env-name "LinearNavigationVisualOriginal-v2" 
            --algo ppo --use-gae --lr 1.15e-04 --clip-param 0.1 
            --value-loss-coef 0.5 --num-processes 4
            --num-steps 5000 --num-mini-batch 4 --log-interval 1 
            --use-linear-lr-decay --entropy-coef 0.01 
            --recurrent-policy --num-env-steps={num_env_steps}
            --save-interval=40 --gamma 0.99 
            --hidden-size={n_neurons}
            --dropout={dropout}
            --weight-decay={weight_decay}
            --l1={l1}
            --l1-rate={l1_rate}
            --l2-rate={l2_rate}
            --recurrent-dropout={recurrent_dropout}
            {"--no-cuda" if not cuda else ""}
            --save-dir={save_dir}'''.replace('\n', '')

    with open(model_path / 'command.txt', 'w') as f:
        print(f'Saved command: {command}')
        f.write(command)

    t0 = time.time()
    subprocess.check_call(command, shell=True)
    t1 = time.time()

    td = t1 - t0
    print(f'Finished in {td} seconds')
    with open(model_path / 'training.done', 'w') as f:
        f.write(str(td))
        f.write('\n')

