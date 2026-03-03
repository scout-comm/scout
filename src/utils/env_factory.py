# Builds Battle (MAgent) and Pursuit (PettingZoo SISL) envs. Battle uses ExpoComm submodule; Pursuit uses src.wrappers.
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
from src.wrappers.pursuit_wrappers import _PursuitWrapper


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
    elif cfg.task == "pursuit_base":
        env = _PursuitWrapper(
            map_name=getattr(cfg, "pursuit_map_name", "pursuit_v3"),
            seed=cfg.seed,
            max_cycles=cfg.max_cycles,
            x_size=getattr(cfg, "x_size", getattr(cfg, "map_size", 16)),
            y_size=getattr(cfg, "y_size", getattr(cfg, "map_size", 16)),
            n_pursuers=getattr(cfg, "n_pursuers", 64),
            n_evaders=getattr(cfg, "n_evaders", 12),
            obs_range=getattr(cfg, "obs_range", 7),
            surround=True,
            n_catch=getattr(cfg, "n_catch", 4),
            local_ratio=getattr(cfg, "local_ratio", 1.0),
            freeze_evaders=getattr(cfg, "freeze_evaders", False),
        )
    else:
        raise ValueError(
            f"Unknown task: {cfg.task}. Supported: battle_* and pursuit_base only."
        )
    return env
