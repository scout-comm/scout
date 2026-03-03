# Builds Battle and AdvPursuit envs only. Requires ExpoComm in third_party (see README).
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
EXPOCOMM_SRC = _repo_root / "third_party" / "ExpoComm" / "src"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(EXPOCOMM_SRC) not in sys.path:
    sys.path.insert(0, str(EXPOCOMM_SRC))

from third_party.ExpoComm.src.envs.battle_wrappers import (
    Battle_w_PretrainedOpp,
    _BattleWrapper,
)
from third_party.ExpoComm.src.envs.adv_pursuit_wrappers import (
    AdvPursuit_w_PretrainedOpp,
    _AdvPursuitWrapper,
)


def make_env(cfg):
    if cfg.task == "battle_pretrained":
        env = Battle_w_PretrainedOpp(
            map_name=cfg.map_name,
            map_size=cfg.map_size,
            max_cycles=cfg.max_cycles,
            minimap_mode=cfg.minimap_mode,
            seed=cfg.seed,
            pretrained_ckpt=cfg.pretrained_filename_battle,
            global_reward=cfg.global_reward,
        )
    elif cfg.task == "battle_base":
        env = _BattleWrapper(
            map_name=cfg.map_name,
            map_size=cfg.map_size,
            max_cycles=cfg.max_cycles,
            minimap_mode=cfg.minimap_mode,
            seed=cfg.seed,
        )
    elif cfg.task == "advpursuit_pretrained":
        env = AdvPursuit_w_PretrainedOpp(
            map_name="adversarial_pursuit_view8",
            map_size=cfg.map_size,
            max_cycles=cfg.max_cycles,
            minimap_mode=cfg.minimap_mode,
            seed=cfg.seed,
            pretrained_ckpt=cfg.pretrained_filename_advpursuit,
            global_reward=cfg.global_reward,
        )
    elif cfg.task == "advpursuit_base":
        env = _AdvPursuitWrapper(
            map_name="adversarial_pursuit_view8",
            map_size=cfg.map_size,
            max_cycles=cfg.max_cycles,
            minimap_mode=cfg.minimap_mode,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown task: {cfg.task}. This release supports battle_* and advpursuit_* only.")
    return env
