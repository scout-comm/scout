# SCOUT: Structured Communication with Learned Grouping in Multi-Agent Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-blue.svg)](https://github.com/scout-comm/scout)

**Website:** [https://scout-comm.github.io](https://scout-comm.github.io)

PyTorch implementation of SCOUT for multi-agent environments with learned communication and group-based recipient selection. This repository contains the core training and evaluation code for Battle and Adversarial Pursuit (MAgent); comparison scripts and experiment runners are not included.

---

## Citation

If you use this code or find it helpful, please cite our paper:

```bibtex
@inproceedings{scout2025rlc,
  title     = {SCOUT: Structured Communication with Learned Grouping in Multi-Agent Reinforcement Learning},
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

In many cooperative multi-agent settings, agents benefit from communicating with a subset of teammates rather than broadcasting to everyone. SCOUT learns both *whom* to talk to (via a soft grouping policy over agent descriptors) and *what* to send (via a shared descriptor and message pool), using a centralized PPO with group-aware value baselines and optional counterfactual advantages for communication actions. The implementation supports Battle and Adversarial Pursuit environments from the [ExpoComm](https://github.com/LXXXXR/ExpoComm) codebase (ICLR 2025).

## Key Features

- **Learned grouping:** Soft assignments over a small number of groups, with balance and edge-utility terms.
- **Descriptor + mailbox:** Per-agent descriptors feed into grouping and message content; a mailbox aggregator delivers messages to recipients.
- **Group-aware critic:** Centralized value function conditions on group membership for better baselines.
- **Optional counterfactual comm advantages:** CommCritic estimates value of sender–recipient pairs for refining send/recv policy gradients.
- **Battle & AdvPursuit:** Compatible with ExpoComm’s MAgent wrappers (with or without pretrained opponents).

## Dependencies

- Python 3.8+
- PyTorch 1.12+
- NumPy, TensorBoard
- PettingZoo 1.12+, Supersuit (for MAgent envs)
- [ExpoComm](https://github.com/LXXXXR/ExpoComm) env files (Battle, Adversarial Pursuit) — added as a submodule below

See `requirements.txt` and the ExpoComm repo for exact versions.

## Installation

### 1. Clone the repository and ExpoComm submodule

Battle and AdvPursuit environments use wrappers from [ExpoComm](https://github.com/LXXXXR/ExpoComm). Clone with the submodule so `third_party/ExpoComm` is populated:

```bash
git clone --recurse-submodules https://github.com/scout-comm/scout.git
cd scout
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

The submodule points to `https://github.com/LXXXXR/ExpoComm`. You should see `third_party/ExpoComm/src` with `envs/battle_wrappers.py` and `envs/adv_pursuit_wrappers.py`.

### 2. Python environment

```bash
pip install -r requirements.txt
```

Install MAgent and PettingZoo as in ExpoComm’s README (e.g. `pip install magent==0.1.14 pettingzoo==1.12.0` and copy their `env/battle_v3_view7.py` and `env/adversarial_pursuit_view8_v3.py` into your PettingZoo `magent` package).

### 3. Pretrained opponents (optional)

For `battle_pretrained` and `advpursuit_pretrained`, the wrappers expect pretrained checkpoint filenames (e.g. `battle.pt`). Place those where the ExpoComm env expects them, or use `battle_base` / `advpursuit_base` for training without opponents.

## Repository layout

| Path | Description |
|------|-------------|
| `src/algos/scout/` | PPO, buffers, descriptor, grouping, schedules; `config.py`, `checkpoint.py`, `comm_critic.py` |
| `src/envs/` | `expocomm_adapter.py`, `pettingzoo_wrappers.py` |
| `src/utils/` | `env_factory.py` (Battle/AdvPursuit), `mailbox_aggregator.py`, `eval_utils.py` |
| `src/trainers/scout_trainer.py` | Main training entrypoint |
| `configs/` | Example YAML (optional; main config is `TrainCfg` in code) |
| `third_party/ExpoComm` | Submodule — [LXXXXR/ExpoComm](https://github.com/LXXXXR/ExpoComm) |

## Usage

From the repository root:

```bash
python -m src.trainers.scout_trainer
```

Defaults use `task="battle_pretrained"`. To change task or scale, edit `TrainCfg` in `src/algos/scout/config.py` or at the bottom of `src/trainers/scout_trainer.py` (e.g. `cfg.task = "advpursuit_pretrained"`, `cfg.map_size = 50`).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ExpoComm](https://github.com/LXXXXR/ExpoComm) (LXXXXR) for MAgent Battle and Adversarial Pursuit wrappers and environment setup.
- EPyMARL and the MAgent / PettingZoo ecosystems for base infra.
