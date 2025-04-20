Please see [this repository](https://github.com/toohuman/NEAD) for a more up-to-date implementation and a less standalone codebase for the neuroevolution of any dynamics.

# AntDynamics-Gymnasium

[![arXiv](https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge)](https://arxiv.org/abs/2406.13147)

We introduce a simulation environment to facilitate research into emergent collective behaviour, with a focus on replicating the dynamics of ant colonies. By leveraging real-world data, the environment simulates a target ant trail that a controllable agent must learn to replicate, using sensory data observed by the target ant. This work aims to contribute to the neuroevolution of models for collective behaviour, focusing on evolving neural architectures that encode domain-specific behaviours in the network topology. By evolving models that can be modified and studied in a controlled environment, we can uncover the necessary conditions required for collective behaviours to emerge. We hope this environment will be useful to those studying the role of interactions in emergent behaviour within collective systems.

The simulation environment implements the [Gymnasium API](https://github.com/Farama-Foundation/Gymnasium) to interface with common off-the-shelf reinforcement learning algorithms, though this can be extended for novel implementations. Specifically, we are interested in neuroevolution to discover features of the resulting topologies. As such, this repository also includes a working implementation of NEAT/WANNs, with fixes to transition to Gymnasium instead of the old Gym API.

## Dependencies

Core algorithm tested with:

- Python 3.11.5

- NumPy 1.26.0 (`pip install numpy`)

- Pygame 2.5.2 (`pip install pygame`)

- mpi4py 3.1.4 (`pip install mpi4py`)

- Gymnasium 0.28.1 (`pip install gymnasium` -- installation details [here](https://github.com/Farama-Foundation/Gymnasium)

### Citation
For attribution in academic contexts, please cite this work as

```
@article{antdyn2024,
  author = {Michael Crosscombe and Ilya Horiguchi and Norihiro Maruyama and Shigeto Dobata and Takashi Ikegami},
  title  = {A Simulation Environment for the Neuroevolution of Ant Colony Dynamics},
  eprint = {https://direct.mit.edu/isal/proceedings/isal2024/36/92/123535},
  url    = {https://github.com/toohuman/AntDynamics-Gymnasium},
  note   = "\url{https://github.com/toohuman/AntDynamics-Gymnasium}",
  year   = {2024}
}
```
