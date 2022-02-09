# Deconstructing The Inductive Biases of Hamiltonian Neural Networks
<p align="center">
  <img src="/assets/figure1.png" width=900>
</p>
This repo contains the implementation and the experiments for the paper 

[Deconstructing The Inductive Biases of Hamiltonian Neural Networks](https://openreview.net/forum?id=EDeVYpT42oS)
by [Nate Gruver](https://ngruver.github.io/), [Marc Finzi](https://mfinzi.github.io/), [Sam Stanton](https://samuelstanton.github.io/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/). 

# Code

Our was constructed from
https://github.com/mfinzi/constrained-hamiltonian-neural-networks.git

# Installation instructions

### Pip
```bash
git clone https://github.com/ngruver/decon-hnn.git
cd decon-hnn
pip install -r requirements.txt
```
### Conda
```bash
git clone https://github.com/ngruver/decon-hnn.git
cd decon-hnn
conda env create -f decon-hnn.yml
```

# View experimental runs and recreate paper figures

All figures from the paper can be recreated by running the notebooks

```
energy_plots.ipynb
bar_plots.ipynb
```

These notebooks process data from the following wandb sweeps

```
https://wandb.ai/ngruver/physics-uncertainty-exps/sweeps
https://wandb.ai/samuelstanton/physics-uncertainty-exps/sweeps
```

All experimental data, and the associated configurations, are contained in these sweeps.

# Train pendulum models

You can train the models ``NN`` (NODE), ``MechanicsNN`` (NODE + SO), and ``HNN`` using the ``model_type`` option as shown below.

```
python toy_systems.py --system_type "ChainPendulum" --model_type "NN"
python toy_systems.py --system_type "ChainPendulum" --model_type "MechanicsNN"
python toy_systems.py --system_type "ChainPendulum" --model_type "HNN"
```

The other systems, with and without friction, can be specified as ``SpringPendulum``, ``FrictionChainPendulum``, and ``FrictionSpringPendulum``. 

# Train Mujoco models

To train models on mujoco, you must first download our saved mujoco trajectories with full state and velocity.

### Mac

```
brew install gdrive
gdrive download 1Vdf8rjPXabfMaCouNfqUYf0ifDW3qAU2 --recursive
mv full_state_mujoco_trajs data
```

### Linux

```
pip install gshell
gshell init
gshell download --with-id '1Vdf8rjPXabfMaCouNfqUYf0ifDW3qAU2' --recursive
mv full_state_mujoco_trajs data
```

Once the data has been downloaded, ``NODE``, ``CoupledNODE``(NODE + SO), and ``MixtureHNN`` (SymODEN) models can be trained as shown below.

```
python mujoco.py --model_type "NODE" --task "HopperFull-v0"
python mujoco.py --model_type "CoupledNODE" --task "HopperFull-v0"
python mujoco.py --model_type "MixtureHNN" --task "HopperFull-v0"
```

The other mujoco tasks included in the paper can be specified as ``SwimmerFull-v0`` and     ``HalfCheetahFull-v0``

# Citation

If you find our work helpful, please cite it with
```bibtex
@inproceedings{
  gruver2022deconstructing,
  title={Deconstructing the Inductive Biases of Hamiltonian Neural Networks},
  author={Nate Gruver and Marc Anton Finzi and Samuel Don Stanton and Andrew Gordon Wilson},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=EDeVYpT42oS}
}
```

