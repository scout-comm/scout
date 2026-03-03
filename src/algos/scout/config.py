# Training config for SCoUT. Kept in one place so the main loop stays readable.
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from src.algos.scout.centralized_ppo import PPOCfg
from src.algos.scout.descriptor import DescriptorConfig
from src.algos.scout.grouping import GroupingConfig


def _default_desc():
    return DescriptorConfig(
        obs_dim=0,
        obs_proj_dim=64,
        include_hidden=True,
        include_time_frac=False,
        include_budget_frac=False,
        include_progress=False,
        include_msg_pool=False,
        normalize_obs=True,
        normalize_msgs=True,
        ortho_coef=0.0,
        nce_temperature=0.1,
    )


def _default_grp():
    return GroupingConfig(
        d_in=0,
        m_groups=4,
        gumbel_tau=0.7,
        dropout=0.0,
        use_prototypes=True,
        logit_scale=3.0,
    )


@dataclass
class TrainCfg:
    task: str = "battle_pretrained"
    pursuit_map_name: str = "pursuit_v3"
    n_pursuers: int = 40
    n_evaders: int = 16
    obs_range: int = 7
    n_catch: int = 4
    surround: bool = True
    local_ratio: float = 1.0
    freeze_evaders: bool = False

    map_name: str = "battle_view7"
    map_size: int = 45
    max_cycles: int = 500
    seed: int = 0
    global_reward: bool = False
    minimap_mode: bool = True

    imp_map_name: str = "struct_c_100"
    imp_struct_type: str = "struct"
    imp_n_comp: int = 50
    imp_discount: float = 0.95
    imp_env_correlation: bool = True
    imp_campaign_cost: bool = True

    pretrained_filename_battle: str = "battle.pt"

    k_macro: int = 10
    m_groups: int = 8
    init_budget: int = 10
    comm_penalty_kappa: float = 0
    msg_loss_coef: float = 1e-2
    hidden: int = 64
    ppo: PPOCfg = field(default_factory=PPOCfg)

    desc: DescriptorConfig = field(default_factory=_default_desc)
    grp: GroupingConfig = field(default_factory=_default_grp)

    rollout_T: int = 2048
    iters: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 10
    lr_grouping: float = 3e-4
    save_dir: str = "runs/scout_new/checkpoints"
    ckpt_every_iter: int = 100
    ckpt_every_iter_latest: int = 10
    ckpt_every_iter_snapshot: int = 200
    resume_ckpt: str = ""
    tb_resume_dir: str = ""
    resume: bool = False
    ret_smooth_ep: int = 100
    tb_dir: str = "runs/scout_new/tb"
    log_groups_every: int = 25
    eval_episodes: int = 5
    eval_every_global_steps: int = 40000

    ablation_no_grouping: bool = False
    ablation_no_counterfactual: bool = False
    ablation_no_comm: bool = True
