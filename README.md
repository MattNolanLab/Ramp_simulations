


## Installation instructions (using conda)

First create a new conda environment and activate it 

```bash
conda create -n tennant2022 python=3.7
conda activate tennant2022
```

Install the VR/linear navigation task [gym](https://github.com/openai/gym)

```bash
pip3 install -e gym-linearnavigation
```

Install pytorch 1.10.0 (instructions may depend on system configuration, see [docs](https://pytorch.org/get-started/previous-versions/))

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the remaining dependencies
```bash
pip install opencv-python==4.5.3.56 scipy==1.7.3 stable-baselines3 pybullet matplotlib h5py
```
## Usage

Train a model using the same parameters as the paper:

```bash
python3 experiments/linearnavigation_param_sweep_visual_paper.py
```

Checkpoints (model weights) will be saved to `results/linearnavigation_visual_paper`

The neuron activity can be saved (to "--output-file [location]") by running one of the checkpoints using the below command:

```bash
python3 models/pytorch-a2c-ppo-acktr-gail/enjoy.py --model-path results/linearnavigation_visual_paper/models/<path_to_checkpoint>.pt --env-name LinearNavigationVisualOriginal-v2 --n-episodes=50 --output-path results/<path_to_output>
```

The above activity data can be used to classify the ramps (using the R scripts in the in vivo codebase) and recreate Figure 6 (figures/Figure_6.Rmd)

To reproduce the perturbation experiments in Figure 7, the activity of readout or recurrent neurons can be clamped or blocked using the `--block-recurrent`, `--block-readout`, `--recurrent-override-values`, `--readout-override-values` arguments combined with the previous enjoy.py script. These can be used to recreate Figure 7 (figures/Figure_7.Rmd)





