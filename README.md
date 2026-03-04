# SCoUT: Structured Communication with Learned Grouping in Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-blue.svg)](https://github.com/scout-comm/scout)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/scout-comm/scout)
[![codecov](https://codecov.io/gh/scout-comm/scout/graph/badge.svg)](https://codecov.io/gh/scout-comm/scout)

**Website:** [https://scout-comm.github.io](https://scout-comm.github.io)

PyTorch implementation of SCoUT for multi-agent environments with learned communication and group-based recipient selection. This repository contains the core training and evaluation code for **MAgent Battle** and **PettingZoo Pursuit (SISL)**; comparison scripts and experiment runners are not included.

---

## Citation

If you use this code or find it helpful, please cite our paper:

```bibtex
@inproceedings{scout2025rlc,
  title     = {SCoUT: Structured Communication with Learned Grouping in Multi-Agent Reinforcement Learning},
  author    = {Anonymous},
  booktitle = {Reinforcement Learning Conference (RLC)},
  year      = {2025},
  url       = {https://github.com/scout-comm/scout},
  note      = {Under Review}
}
```

*BibTeX will be updated upon publication.*

---

## Overview

In many cooperative multi-agent settings, agents benefit from communicating with a subset of teammates rather than broadcasting to everyone. SCoUT learns both *whom* to talk to (via a soft grouping policy over agent descriptors) and *what* to send (via a shared descriptor and message pool), using a centralized PPO with group-aware value baselines and optional counterfactual advantages for communication actions. The implementation supports:

- **MAgent Battle** — via [ExpoComm](https://github.com/LXXXXR/ExpoComm) battle wrappers (with or without pretrained opponents).
- **PettingZoo Pursuit (SISL)** — the standard `pursuit_v3` environment, wrapped in `src/wrappers/pursuit_wrappers.py` for fixed-shape obs/actions and compatibility with the trainer.

We do *not* use the Adversarial Pursuit (MAgent) environment; Pursuit here refers only to the SISL Pursuit environment.

## Key Features

- **Learned grouping:** Soft assignments over a small number of groups, with balance and edge-utility terms.
- **Descriptor + mailbox:** Per-agent descriptors feed into grouping and message content; a mailbox aggregator delivers messages to recipients.
- **Group-aware critic:** Centralized value function conditions on group membership for better baselines.
- **Optional counterfactual comm advantages:** CommCritic estimates value of sender–recipient pairs for refining send/recv policy gradients.
- **Battle & Pursuit (SISL):** Battle uses ExpoComm’s MAgent wrappers; Pursuit uses our PettingZoo SISL wrapper.

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- NumPy, TensorBoard
- PettingZoo 1.12+, Supersuit (for Pursuit and for MAgent Battle)
- [ExpoComm](https://github.com/LXXXXR/ExpoComm) — submodule for **Battle** envs only (see below)

See `requirements.txt` and the ExpoComm repo for MAgent/PettingZoo versions.

## Installation

### 1. Clone the repository and ExpoComm submodule

**Battle** uses wrappers from [ExpoComm](https://github.com/LXXXXR/ExpoComm). Clone with the submodule so `third_party/ExpoComm` is populated:

```bash
git clone --recurse-submodules https://github.com/scout-comm/scout.git
cd scout
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

You should see `third_party/ExpoComm/src` with `envs/battle_wrappers.py`. The submodule is only required for Battle; **Pursuit (SISL)** uses PettingZoo directly and the wrappers in `src/wrappers/`.

### 2. Python environment

```bash
pip install -r requirements.txt
```

- **For Battle:** Install MAgent and PettingZoo as in ExpoComm’s README (e.g. `pip install magent==0.1.14 pettingzoo==1.12.0` and copy ExpoComm’s `env/battle_v3_view7.py` into your PettingZoo `magent` package).
- **For Pursuit:** Install PettingZoo SISL and Supersuit (e.g. `pip install pettingzoo supersuit`); the Pursuit env is `pettingzoo.sisl.pursuit_v3`.

### 3. Pretrained opponents (Battle only, optional)

For `battle_pretrained`, the wrapper expects a pretrained checkpoint filename (e.g. `battle.pt`). Place it where the ExpoComm env expects it, or use `battle_base` for training without opponents.

## Repository layout

| Path | Description |
|------|-------------|
| `src/algos/scout/` | PPO, buffers, descriptor, grouping, schedules; `config.py`, `checkpoint.py`, `comm_critic.py` |
| `src/envs/` | `expocomm_adapter.py`, `pettingzoo_wrappers.py` |
| `src/utils/` | `env_factory.py` (Battle + Pursuit), `mailbox_aggregator.py`, `eval_utils.py` |
| `src/wrappers/` | `pursuit_wrappers.py` (PettingZoo SISL Pursuit), `magent.py`, `multiagentenv.py` |
| `src/trainers/scout_trainer.py` | Main training entrypoint |
| `configs/` | Example YAML (optional; main config is `TrainCfg` in code) |
| `third_party/ExpoComm` | Submodule — [LXXXXR/ExpoComm](https://github.com/LXXXXR/ExpoComm) (Battle only) |

## Usage

From the repository root:

```bash
python -m src.trainers.scout_trainer
```

Defaults use `task="battle_pretrained"`. To run **Pursuit (SISL)** instead, set `cfg.task = "pursuit_base"` in `src/algos/scout/config.py` or at the bottom of `src/trainers/scout_trainer.py`, and set pursuit options (e.g. `cfg.n_pursuers`, `cfg.n_evaders`, `cfg.map_size` for grid size).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ExpoComm](https://github.com/LXXXXR/ExpoComm) (LXXXXR) for MAgent Battle wrappers and environment setup.
- PettingZoo and the SISL Pursuit environment.
- EPyMARL and the MAgent / PettingZoo ecosystems for base infra.
