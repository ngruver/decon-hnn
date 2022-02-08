# Deconstructing The Inductive Biases of Hamiltonian Neural Networks
<p align="center">
  <img src="/assets/figure1.pdf" width=900>
</p>
This repo contains the implementation and the experiments for the paper 

[Deconstructing The Inductive Biases of Hamiltonian Neural Networks](https://openreview.net/forum?id=EDeVYpT42oS)
by [Nate Gruver](https://ngruver.github.io/), [Marc Finzi](https://mfinzi.github.io/), [Sam Stanton](https://samuelstanton.github.io/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/). 

<!-- # Code
Our code in the `biases` directory relies on some publically available codebases which we package together
as a conda environment. [![Code Climate maintainability](https://api.codeclimate.com/v1/badges/a99a88d28ad37a79dbf6/maintainability)](https://codeclimate.com/github/mfinzi/hamiltonian-biases/maintainability) [![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://xkcd.com/54/)

# Installation instructions
Install PyTorch>=1.0.0

(Optional) Create a wandb account for experiment tracking
## Pip
```bash
git clone https://github.com/mfinzi/constrained-hamiltonian-neural-networks.git
cd constrained-hamiltonian-neural-networks
pip install -e .
```
## Conda
```bash
git clone https://github.com/mfinzi/constrained-hamiltonian-neural-networks.git
cd constrained-hamiltonian-neural-networks
conda env create -f conda_env.yml
pip install ./
```

# Train Models
We have implemented a variety of challenging benchmarks for modeling physical dynamical systems such as ``ChainPendulum``, ``CoupledPendulum``,``MagnetPendulum``,``Gyroscope``,``Rotor`` which can be selected with the ``--body-class`` argument.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081999-eb897a80-fdcd-11ea-8e29-c676d4e25f64.PNG" width=1000>
</p>

You can run our models ``CHNN`` and ``CLNN`` as well as the baseline ``NN`` (NeuralODE), ``DeLaN``, and ``HNN`` models with the ``network-class`` argument as shown below.

```
python pl_trainer.py --network-class CHNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class CLNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class HNN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class DeLaN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
python pl_trainer.py --network-class NN --body-class Gyroscope --wandb-project "YOUR WANDB PROJECT"
```

Our explicitly constrained ``CHNN`` and ``CLNN`` outperform the competing methods by several orders of magnitude across the different benchmarks as shown below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/94081992-e75d5d00-fdcd-11ea-9df0-576af6909944.PNG" width=1000>
</p> -->

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

