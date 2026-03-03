# src/utils/eval_utils.py
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
from src.envs.expocomm_adapter import ExpoCommAdapter, ExpoCommAdapterCfg
from src.utils.env_factory import make_env
from src.utils.mailbox_aggregator import MailboxAttention


@torch.no_grad()
def run_eval(
    ppo,
    descriptor,
    cfg,
    *,
    episodes: int = 10,
    force_no_comm: bool = False,
    stochastic: bool = True,
    grouping=None,  # NEW: pass GroupingPolicy or None
    use_grouping: bool = False,  # NEW: toggle grouping during eval
    grp_tau: float = None,  # NEW: override τ (defaults to cfg.grp.gumbel_tau)
    episode_seeds: Optional[Sequence[int]] = None,
):
    # fresh env so training state isn't perturbed
    env_eval = make_env(cfg)
    adz = ExpoCommAdapter(
        env_eval, ExpoCommAdapterCfg(comm_radius=None, use_alive_heuristic=False)
    )

    device = next(ppo.parameters()).device
    attn_pool = MailboxAttention(dim=descriptor.cfg.msg_proj_dim).to(device)
    ppo.policy.eval()
    descriptor.eval()
    if grouping is not None:
        grouping.eval()

    if grp_tau is None:
        grp_tau = getattr(getattr(cfg, "grp", object()), "gumbel_tau", 1.0)

    wins, lens, rets_env = 0, [], []
    wins_list = []
    red_alive_list, blue_alive_list = [], []
    for ep in range(episodes):
        seed_arg = None if episode_seeds is None else int(episode_seeds[ep])
        obs_np, state_np, env_mask_np, _ = adz.reset(seed=seed_arg)
        info_eval = env_eval.get_env_info()
        A = int(info_eval["n_agents"])
        D_MSG = descriptor.cfg.msg_proj_dim

        obs_t = torch.from_numpy(obs_np).to(device)
        state_t = torch.from_numpy(state_np).to(device)
        rmask = torch.from_numpy(env_mask_np).to(device)
        h_t = ppo.policy.init_hidden(A, device)
        inbox = torch.zeros((A, D_MSG), device=device)

        ep_ret_env, done_all, ep_steps = 0.0, False, 0
        while not done_all:
            # single pass through descriptor
            heads = descriptor.heads(
                {"obs": obs_t, "hidden": h_t}, device=device, update_norm=False
            )
            z_msg = heads["z_msg"]  # (A, D_MSG)
            z_grp = heads.get("z_grp", None)  # (A, d_grp) if available

            # choose grouping affinity if requested
            G_soft = None
            if use_grouping and (grouping is not None) and (z_grp is not None):
                grp_cache = grouping.sample(z_grp, tau=grp_tau)
                G_soft = grp_cache["G"]  # (A, A) soft affinity

            # augmented obs (zero inbox when force_no_comm)
            obs_aug = torch.cat(
                [obs_t, (0 * inbox if force_no_comm else inbox)], dim=-1
            )

            if stochastic:
                out = ppo.act(obs_aug, h_t, G_soft=G_soft, recv_mask=rmask)
                env_a, send_a, recv_a, h_next = (
                    out["env_action"],
                    out["send_action"],
                    out["recv_action"],
                    out["h_out"],
                )
            else:
                out = ppo.evaluate(obs_aug, h_t, G_soft=G_soft, recv_mask=rmask)
                env_a = out["env_logits"].argmax(dim=-1)
                send_a = out["send_logits"].argmax(dim=-1)
                recv_a = out["recv_logits"].argmax(dim=-1)
                h_next = out["h_out"]

            if force_no_comm:
                send_a = torch.zeros_like(send_a)
                recv_a = torch.zeros_like(recv_a)

            # step env
            obs_np, state_np, rew_np, dones_np, env_mask_np, info = adz.step(
                env_actions=env_a.cpu().numpy(),
                send=send_a.cpu().numpy(),
                recv=recv_a.cpu().numpy(),
            )
            obs_next = torch.from_numpy(obs_np).to(device)
            state_next = torch.from_numpy(state_np).to(device)
            rmask_next = torch.from_numpy(env_mask_np).to(device)

            # route messages only if comm allowed
            if (not force_no_comm) and (send_a.sum().item() > 0):
                i_idx = (
                    (send_a == 1).nonzero(as_tuple=False).squeeze(-1)
                )  # senders at t
                j_idx = recv_a[i_idx]  # chosen recipients

                if i_idx.numel() > 0:
                    # respect environment receive mask exactly like training
                    allowed = rmask[i_idx, j_idx].bool()
                    i_idx = i_idx[allowed]
                    j_idx = j_idx[allowed]

                    if i_idx.numel() > 0:
                        # param-free attention in z-space; q=z_j, k=v=z_i (same as training)
                        inbox = attn_pool(
                            z_msg_all=z_msg,
                            recv_query_all=z_msg,  # q=z_j
                            i_idx=i_idx,
                            j_idx=j_idx,
                            A=A,
                        )
                    else:
                        inbox.zero_()
                else:
                    inbox.zero_()
            else:
                inbox.zero_()

            # bookkeeping
            ep_steps += 1
            ep_ret_env += float(np.mean(rew_np))
            obs_t, state_t, h_t, rmask = obs_next, state_next, h_next, rmask_next
            done_all = bool(np.all(dones_np))
            if done_all:
                raw = info.get("raw_info", info)  # works with or without adapter
                # print(f'[eval] info: {raw}')
                red_team_alives = raw.get("red_team_alives", 0)
                blue_team_alives = raw.get("blue_team_alives", 0)
                wins += int(bool(raw.get("red_team_win", False)))
                red_alive_list.append(red_team_alives)
                blue_alive_list.append(blue_team_alives)
                wins_list.append(int(bool(raw.get("red_team_win", False))))
                lens.append(int(raw.get("episode_length", ep_steps)))
                rets_env.append(ep_ret_env)

    wins_arr = np.array(wins_list, dtype=float)
    red_arr = np.array(red_alive_list, dtype=float)
    blue_arr = np.array(blue_alive_list, dtype=float)
    lens_arr = np.array(lens, dtype=float)
    rets_arr = np.array(rets_env, dtype=float)

    return {
        "red_team_alives": float(red_arr.mean()) if red_arr.size else 0.0,
        "blue_team_alives": float(blue_arr.mean()) if blue_arr.size else 0.0,
        "win_rate": float(wins_arr.mean()) if wins_arr.size else 0.0,
        "ret_env": float(rets_arr.mean()) if rets_arr.size else 0.0,
        "len": float(lens_arr.mean()) if lens_arr.size else 0.0,
        # per-episode values for comm-effect analysis:
        "per_episode_red_alive": red_alive_list,
        "per_episode_blue_alive": blue_alive_list,
        "per_episode_win": wins_list,
        "per_episode_ret": rets_env,
        "per_episode_len": lens,
    }
