# MIT License
# src/scout_trainer.py
from __future__ import annotations
import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# --- Fix for NumPy ≥1.24 removing deprecated aliases used by magent ---
if not hasattr(np, "bool"):
    np.bool = bool
    np.int = int
    np.float = float
    np.object = object
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import json
import types
from collections import deque
import time
import copy
from src.algos.scout.buffers import BufferSpec, RolloutBuffer
from src.algos.scout.descriptor import DescriptorConfig, DescriptorBuilder
from src.algos.scout.grouping import GroupingConfig, GroupingPolicy
from src.algos.scout.centralized_ppo import PPOCfg, CentralizedPPO, compute_gae
from src.envs.expocomm_adapter import ExpoCommAdapter, ExpoCommAdapterCfg
from src.algos.scout.schedules import LinearSchedule, CosineSchedule
from src.utils.env_factory import make_env
from src.utils.eval_utils import run_eval
from src.utils.mailbox_aggregator import MailboxAttention
from src.algos.scout.config import TrainCfg
from src.algos.scout.checkpoint import (
    resolve_resume_path,
    make_ckpt,
    save_latest,
    save_snapshot,
    load_ckpt,
)
from src.algos.scout.comm_critic import CommCritic


# we’ll need these two helpers to build negatives etc.
def _one_hot(idx: torch.Tensor, num: int) -> torch.Tensor:
    out = torch.zeros(idx.shape[0], num, device=idx.device, dtype=torch.float32)
    out.scatter_(1, idx.view(-1, 1), 1.0)
    return out



def train(cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    # ---------- Checkpoint dirs / resume ----------
    # dirs
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.save_dir) / f"{cfg.task}_{run_id}"
    latest_dir = Path(cfg.save_dir) / f"{cfg.task}_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    tb_run_dir = Path(cfg.tb_dir) / f"{cfg.task}_{run_id}"

    # ---------- Episodic return trackers ----------
    ret_env_window = deque(maxlen=cfg.ret_smooth_ep)  # before comm penalty
    ret_net_window = deque(maxlen=cfg.ret_smooth_ep)  # after comm penalty
    len_window = deque(maxlen=cfg.ret_smooth_ep)
    ep_ret_env = 0.0
    ep_ret_net = 0.0
    ep_len = 0
    completed_eps = 0

    # ---------- Env + adapter ----------
    raw_env = make_env(cfg)
    env_info = raw_env.get_env_info()
    if cfg.task == "imp_struct":
        adz = raw_env  # already IMPAdapter
    else:
        adz = ExpoCommAdapter(
            raw_env,
            ExpoCommAdapterCfg(
                comm_radius=None,  # or a float to restrict comm by distance
                use_alive_heuristic=False,  # wrappers already zero dead slots
            ),
        )

    # ent_action_sched = CosineSchedule(0.02, 0.005, iters=cfg.iters)
    # ent_send_sched   = CosineSchedule(0.01, 0.003, iters=cfg.iters)
    # ent_recv_sched   = CosineSchedule(0.01, 0.003, iters=cfg.iters)
    # lr_sched         = LinearSchedule(3e-4, 1e-4, iters=cfg.iters)
    grp_tau_sched = LinearSchedule(10.0, 0.5, iters=cfg.iters)
    grp_lambda_edge_sched = LinearSchedule(0.0, 1.0, iters=cfg.iters)
    # kappa_sched = LinearSchedule(0.0, cfg.comm_penalty_kappa, iters=int(0.2*cfg.iters))
    # grp_logit_scale_sched = LinearSchedule(3.0, 10.0, iters=cfg.iters)
    # Probe shapes
    obs_np, state_np, env_mask_np, _ = adz.reset()
    info = raw_env.get_env_info()
    A = int(info["n_agents"])
    print(f"Env has {A} agents.")
    obs_dim = int(info["obs_shape"])
    state_dim = int(info["state_shape"])
    n_actions = int(info["n_actions"])

    # ---------- Models ----------
    desc_cfg = cfg.desc
    desc_cfg.obs_dim = obs_dim
    desc_cfg.hidden_dim = cfg.hidden
    # Compute "core" descriptor width (without msg_pool)
    D_MSG = (
        desc_cfg.obs_proj_dim
        + (desc_cfg.hidden_proj_dim if desc_cfg.include_hidden else 0)
        + (1 if desc_cfg.include_time_frac else 0)
        + (1 if desc_cfg.include_budget_frac else 0)
        + (1 if desc_cfg.include_progress else 0)
    )
    # D_MSG = obs_dim + cfg.hidden  # simpler: just obs + hidden
    desc_cfg.msg_dim = D_MSG  # <--- mailbox width == core descriptor
    desc_cfg.msg_proj_dim = D_MSG  # <--- projected msg dim (for grouping & actor)

    descriptor = DescriptorBuilder(desc_cfg).to(device)
    attn_pool = MailboxAttention(dim=desc_cfg.msg_proj_dim).to(device)

    obs_dim_aug = obs_dim + D_MSG  # actor sees obs || mailbox

    # ========== ABLATION: No Grouping ==========
    # Override m_groups to 1 if ablation_no_grouping is set
    effective_m_groups = 1 if cfg.ablation_no_grouping else cfg.m_groups
    if cfg.ablation_no_grouping:
        print("[ABLATION] No Grouping: M=1, G=all-ones (no recipient bias, shared baseline)")

    grp_cfg = cfg.grp
    grp_cfg.d_in = descriptor.cfg.grp_proj_dim  # <--- instead of d_xi
    grp_cfg.m_groups = effective_m_groups
    grouping = GroupingPolicy(grp_cfg).to(device)
    opt_grouping = torch.optim.Adam(grouping.parameters(), lr=cfg.lr_grouping)
    opt_desc = torch.optim.Adam(descriptor.parameters(), lr=cfg.lr_grouping)

    ppo = CentralizedPPO(
        obs_dim=obs_dim_aug,
        state_dim=state_dim,
        n_actions=n_actions,
        n_agents=A,
        n_groups=effective_m_groups,
        hidden=cfg.hidden,
        cfg=cfg.ppo,
    ).to(device)

    # ---------- Buffer ----------
    buf = RolloutBuffer(
        BufferSpec(
            T=cfg.rollout_T,
            A=A,
            obs_dim=obs_dim_aug,
            state_dim=state_dim,
            device=device,
            hidden_dim=cfg.hidden,
            msg_dim=D_MSG,
            z_msg_dim=desc_cfg.msg_proj_dim,
            m_groups=effective_m_groups,
        )
    )

    # ---- CommCritic: agent feature = [ z_msg_dim + incoming_msg_dim + 1(send bit) + m_groups(onehot or soft P) ]
    z_msg_dim = descriptor.cfg.msg_proj_dim
    incoming_dim = D_MSG
    agent_feat_dim = z_msg_dim + incoming_dim + 1 + effective_m_groups

    # ========== ABLATION: No Counterfactual ==========
    if cfg.ablation_no_counterfactual:
        print("[ABLATION] No Counterfactual: using GAE advantages for comm heads (no CommCritic)")
        comm_critic = None
        comm_critic_tgt = None
        opt_comm = None
    else:
        comm_critic = CommCritic(
            state_dim=state_dim, agent_feat_dim=agent_feat_dim, hidden=cfg.hidden
        ).to(device)
        comm_critic_tgt = copy.deepcopy(comm_critic).eval()
        for p in comm_critic_tgt.parameters():
            p.requires_grad_(False)
        opt_comm = torch.optim.Adam(comm_critic.parameters(), lr=5e-4)

    # ========== ABLATION: No Communication ==========
    if cfg.ablation_no_comm:
        print("[ABLATION] No Communication: forcing send=0 at all times")

        # ---------- Resume (after models/opts exist) ----------
    start_it = 1
    if cfg.resume:
        ckpt_path = resolve_resume_path(cfg)
        if ckpt_path.exists():
            global_step, last_it, meta = load_ckpt(
                ckpt_path,
                ppo, grouping, descriptor, device,
                comm_critic=comm_critic,
                comm_critic_tgt=comm_critic_tgt,
                opt_comm=opt_comm,
                opt_grouping=opt_grouping,
                opt_desc=opt_desc,
            )
            start_it = last_it + 1

            # Override per-run directories from checkpoint (keeps per-run TB)
            if "run_dir" in meta:
                run_dir = Path(meta["run_dir"])
            if cfg.tb_resume_dir:
                tb_run_dir = Path(cfg.tb_resume_dir)  # explicit override wins
            elif "tb_run_dir" in meta:
                tb_run_dir = Path(meta["tb_run_dir"])
            if "run_id" in meta:
                run_id = meta["run_id"]

            print(f"[resume] loaded {ckpt_path}")
            print(f"[resume] global_step={global_step}, start_it={start_it}")
            print(f"[resume] run_dir={run_dir}")
            print(f"[resume] tb_run_dir={tb_run_dir}")
        else:
            print(f"[resume] requested but not found: {ckpt_path}. Starting fresh.")
            global_step = 0
            start_it = 1
    else:
        global_step = 0
        start_it = 1

    # ---------- Training ----------
    k_macro = cfg.k_macro
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_run_dir))

    # ppo.cfg.comm_rate_target = 0.5
    # ppo.cfg.comm_rate_coef = 0.1  # no adaptive budget loss for now
    # ppo.cfg.loss_weight_send = 0.0
    # ppo.cfg.loss_weight_recv = 0.0
    # ppo.cfg.entropy_coef_send = 0.0
    # ppo.cfg.entropy_coef_recv = 0.0

    # obs_t = torch.from_numpy(obs_np).to(device)
    # state_t = torch.from_numpy(state_np).to(device)
    # env_mask_t = torch.from_numpy(env_mask_np).to(device)
    # h_t = ppo.policy.init_hidden(A, device)
    # msg_pool_t = torch.zeros((A, D_MSG), device=device)

    for it in range(start_it, cfg.iters + 1):
        print(f"\n=== Iteration {it} ===")
        # reset episode-level things

        obs_t = torch.from_numpy(obs_np).to(device)
        h_t = ppo.policy.init_hidden(A, device)
        state_t = torch.from_numpy(state_np).to(device)
        env_mask_t = torch.from_numpy(env_mask_np).to(device)
        msg_pool_t = torch.zeros(
            (A, D_MSG), device=device
        )  # reset per iteration; also reset on episode reset

        # --- grouping logging accumulators for this iteration ---
        group_sizes_all = []  # list of np arrays [m_groups] collected each macro-block
        sample_group_json = None  # one representative grouping mapping for add_text

        # --- caches for per-step and per-block (for counterfactuals & grouping loss)
        perstep_msg_pool_vis = []  # list of (A, D_MSG)
        perstep_z_msg = []  # list of (A, z_msg_dim)
        perstep_grp_P_tau = (
            []
        )  # (A, M) for each t (repeat the block's P_tau across its k steps)
        perstep_grp_G = []  # (A, A) per t (repeat block's G)

        block_logp_tau = []  # list of (A,) at block starts
        block_P_tau = []  # list of (A, M) at block starts
        block_G = []  # list of (A, A) at block starts
        block_tstarts = []  # list of ints

        t = 0
        while t < cfg.rollout_T:
            # ---------- GROUP ONCE every k ----------
            # Build descriptor heads for grouping/messages (NO msg_pool here)
            heads_group = descriptor.heads(
                {"obs": obs_t, "hidden": h_t}, device=device, update_norm=True
            )
            # Optional: keep xi if you log/inspect it later; z_grp drives grouping; z_msg is the comm content
            xi = heads_group["xi"]  # [A, d_xi] (shared trunk; optional)
            zgrp = heads_group["z_grp"]  # [A, d_grp]
            # zmsg_now = heads_group[
            #     "z_msg"
            # ]  # [A, d_msg]  (message content for this macro block)

            # Sample soft group assignments ONCE per block (consistent τ)
            grp_tau = grp_tau_sched.at(it)
            grp_cache = grouping.sample(zgrp, tau=grp_tau)
            Y_soft_block = grp_cache["y_soft"]  # [A, M] (soft sample used to build G)
            P_tau_block = grp_cache[
                "P_tau"
            ]  # [A, M] (softmax(logits/τ); used by critic)
            G_block = grp_cache["G"]  # [A, A] = Y Y^T (soft affinity)
            logp_tau_blk = grp_cache[
                "logp_grp_tau"
            ]  # [A]    (same-τ log-prob; for grouping PG loss)

            # ========== ABLATION: No Grouping ==========
            # Override G to all-ones (no recipient bias) when M=1
            if cfg.ablation_no_grouping:
                G_block = torch.ones(A, A, device=device)
                # P_tau_block is already (A, 1) uniform since M=1
            # print("after sample:", logp_tau_blk.requires_grad, P_tau_block.requires_grad, G_block.requires_grad)
            # ---- LOGGING: keep your existing group-size logging (uses hard argmax JUST FOR LOGGING)
            gidx = Y_soft_block.argmax(
                dim=-1
            )  # [A]  <-- purely for diagnostics, not used in loss
            sizes = (
                torch.bincount(gidx, minlength=effective_m_groups)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            group_sizes_all.append(sizes)

            if sample_group_json is None:
                sample_group_json = {
                    int(g): [
                        int(i)
                        for i in (gidx == g).nonzero(as_tuple=False).view(-1).tolist()
                    ]
                    for g in range(effective_m_groups)
                }

            # ---- BLOCK-LEVEL CACHES (used later: PPO critic + grouping loss)
            block_logp_tau.append(logp_tau_blk.clone())
            block_P_tau.append(P_tau_block.clone())
            block_G.append(G_block.clone())
            block_tstarts.append(int(t))

            # ---- RECIPIENT CONSTRAINTS FOR THIS BLOCK
            # Keep environment validity as a hard mask; group preference is injected softly via log(G+eps) inside ppo.act(...)
            # recv_mask_block = env_mask_t.clone()
            t_block_start = t

            # ---------- k primitive steps ----------
            k_t = min(k_macro, cfg.rollout_T - t)

            for j in range(k_t):
                # ---- build augmented obs = [raw obs || mailbox]
                recv_mask_step = env_mask_t.clone()
                recv_mask_step.fill_diagonal_(False)  # disallow self as recipient
                obs_aug_t = torch.cat([obs_t, msg_pool_t], dim=-1)
                h_used = h_t
                act = ppo.act(
                    obs_aug_t, h_used, G_soft=G_block, recv_mask=recv_mask_step
                )
                send_a = act["send_action"]
                recv_a = act["recv_action"]
                logp_send = act["logp_send"]

                # ========== ABLATION: No Communication ==========
                # Must override AFTER extracting from act dict
                if cfg.ablation_no_comm:
                    send_a = torch.zeros_like(send_a)
                    # Recompute logp_send for the forced action (send=0)
                    # logp_send should be log P(send=0) under current policy
                    with torch.no_grad():
                        send_logits = act.get("send_logits", None)
                        if send_logits is not None:
                            logp_send = F.log_softmax(send_logits, dim=-1)[:, 0]  # log P(send=0)
                        # If send_logits not available, logp_send stays as-is (less accurate but works)

                # valid senders and allowed recipients
                send_mask_now = send_a == 1
                i_idx = torch.empty(0, dtype=torch.long, device=device)
                j_idx = torch.empty(0, dtype=torch.long, device=device)
                if send_mask_now.any():
                    i_tmp = send_mask_now.nonzero(as_tuple=False).squeeze(-1)  # (Ns,)
                    j_all = recv_a[i_tmp]  # (Ns,)
                    allowed = recv_mask_step[i_tmp, j_all]  # same-group + env validity
                    i_idx = i_tmp[allowed]
                    j_idx = j_all[allowed]  # (Nv,)
                h_next = act["h_out"].detach()

                with torch.no_grad():
                    heads = descriptor.heads(
                        {"obs": obs_t, "hidden": h_used},
                        device=device,
                        update_norm=True,
                    )
                zmsg_now = heads["z_msg"]  # [A, d_msg]

                perstep_msg_pool_vis.append(msg_pool_t.detach().clone())
                perstep_z_msg.append(zmsg_now.detach().clone())
                # step env
                obs_np, state_np, rew_np, dones_np, env_mask_np, _ = adz.step(
                    env_actions=act["env_action"].cpu().numpy(),
                    send=send_a.cpu().numpy(),
                    recv=act["recv_action"].cpu().numpy(),
                )

                rewards = torch.from_numpy(rew_np).to(device)
                dones = torch.from_numpy(dones_np).to(device)
                obs_next = torch.from_numpy(obs_np).to(device)
                state_next = torch.from_numpy(state_np).to(device)
                env_mask_t = torch.from_numpy(env_mask_np).to(device)

                h_t = h_next
                # ---- update mailbox for next step
                with torch.no_grad():
                    msg_pool_next = torch.zeros_like(msg_pool_t)
                    if i_idx.numel() > 0:
                        recv_queries = zmsg_now  # (A, D_MSG)
                        msg_pool_next = attn_pool(
                            zmsg_now, recv_queries, i_idx, j_idx, A
                        )

                # one-step latency: the mailbox we just built becomes visible next step
                msg_pool_t = msg_pool_next
                perstep_grp_P_tau.append(P_tau_block.detach().clone())
                perstep_grp_G.append(G_block.detach().clone())

                # ---- compute net rewards with comm penalty
                # kappa = kappa_sched.at(it)
                rewards_env = rewards
                # rewards_net = rewards_env - kappa * send_a.float()
                rewards_net = rewards_env
                # track episodic returns
                ep_ret_env += rewards_env.mean().item()
                ep_ret_net += rewards_net.mean().item()
                ep_len += 1
                with torch.no_grad():
                    values_t = ppo.value(
                        state_t.unsqueeze(0), P_tau_block.unsqueeze(0)
                    ).squeeze(
                        0
                    )  # (A,)

                had_valid_recv = recv_mask_step.any(dim=-1)
                buf.add_step(
                    obs=obs_aug_t,
                    state=state_t.detach(),
                    actions=act["env_action"],
                    send=send_a,
                    recv=act["recv_action"],
                    logp_env=act["logp_env"].detach(),
                    logp_send=logp_send.detach(),
                    logp_recv=act["logp_recv"].detach(),
                    reward=rewards_net,
                    done=dones,
                    value=values_t,
                    recv_mask=recv_mask_step,
                    had_valid_recv=had_valid_recv,
                    hidden=h_used.detach(),  # for possible auxiliary losses later
                    msg_pool_visible=perstep_msg_pool_vis[-1],
                    z_msg=zmsg_now.detach(),
                    grp_P_tau=P_tau_block.detach(),
                    grp_G=G_block.detach(),
                )
                # print("stored send[t].mean:", buf.send[t].float().mean().item())
                # advance
                obs_t, state_t = obs_next, state_next
                t += 1
                global_step += 1

                if global_step % cfg.eval_every_global_steps == 0:
                    print(f"--- Eval at it {it} (step {global_step}) ---")

                    # Baseline: NO grouping bias
                    res_comm_nogrp = run_eval(
                        ppo,
                        descriptor,
                        cfg,
                        episodes=cfg.eval_episodes,
                        force_no_comm=False,
                        stochastic=True,
                        grouping=None,
                        use_grouping=False,
                    )
                    res_nocomm_nogrp = run_eval(
                        ppo,
                        descriptor,
                        cfg,
                        episodes=cfg.eval_episodes,
                        force_no_comm=True,
                        stochastic=True,
                        grouping=None,
                        use_grouping=False,
                    )

                    writer.add_scalars(
                        "eval_nogrp/red_team_alives",
                        {
                            "comm_on": res_comm_nogrp["red_team_alives"],
                            "no_comm": res_nocomm_nogrp["red_team_alives"],
                        },
                        global_step,
                    )

                    writer.add_scalars(
                        "eval_nogrp/blue_team_alives",
                        {
                            "comm_on": res_comm_nogrp["blue_team_alives"],
                            "no_comm": res_nocomm_nogrp["blue_team_alives"],
                        },
                        global_step,
                    )

                    writer.add_scalars(
                        "eval_nogrp/win_rate",
                        {
                            "comm_on": res_comm_nogrp["win_rate"],
                            "no_comm": res_nocomm_nogrp["win_rate"],
                        },
                        global_step,
                    )
                    writer.add_scalars(
                        "eval_nogrp/ret_env",
                        {
                            "comm_on": res_comm_nogrp["ret_env"],
                            "no_comm": res_nocomm_nogrp["ret_env"],
                        },
                        global_step,
                    )
                    writer.add_scalars(
                        "eval_nogrp/len",
                        {
                            "comm_on": res_comm_nogrp["len"],
                            "no_comm": res_nocomm_nogrp["len"],
                        },
                        global_step,
                    )

                    # With grouping bias ON (use your fixed τ; change grp_tau if you want a different eval τ)
                    res_comm_grp = run_eval(
                        ppo,
                        descriptor,
                        cfg,
                        episodes=cfg.eval_episodes,
                        force_no_comm=False,
                        stochastic=True,
                        grouping=grouping,
                        use_grouping=True,
                        grp_tau=cfg.grp.gumbel_tau,
                    )
                    res_nocomm_grp = run_eval(
                        ppo,
                        descriptor,
                        cfg,
                        episodes=cfg.eval_episodes,
                        force_no_comm=True,
                        stochastic=True,
                        grouping=grouping,
                        use_grouping=True,
                        grp_tau=cfg.grp.gumbel_tau,
                    )

                    writer.add_scalars(
                        "eval_grp/red_team_alives",
                        {
                            "comm_on": res_comm_grp["red_team_alives"],
                            "no_comm": res_nocomm_grp["red_team_alives"],
                        },
                        global_step,
                    )

                    writer.add_scalars(
                        "eval_grp/blue_team_alives",
                        {
                            "comm_on": res_comm_grp["blue_team_alives"],
                            "no_comm": res_nocomm_grp["blue_team_alives"],
                        },
                        global_step,
                    )

                    writer.add_scalars(
                        "eval_grp/win_rate",
                        {
                            "comm_on": res_comm_grp["win_rate"],
                            "no_comm": res_nocomm_grp["win_rate"],
                        },
                        global_step,
                    )
                    writer.add_scalars(
                        "eval_grp/ret_env",
                        {
                            "comm_on": res_comm_grp["ret_env"],
                            "no_comm": res_nocomm_grp["ret_env"],
                        },
                        global_step,
                    )
                    writer.add_scalars(
                        "eval_grp/len",
                        {
                            "comm_on": res_comm_grp["len"],
                            "no_comm": res_nocomm_grp["len"],
                        },
                        global_step,
                    )

                    writer.flush()

                if dones.all():
                    # full episode ended -> reset
                    ret_env_window.append(ep_ret_env)
                    ret_net_window.append(ep_ret_net)
                    len_window.append(ep_len)
                    completed_eps += 1
                    ep_ret_env, ep_ret_net, ep_len = 0.0, 0.0, 0
                    obs_np, state_np, env_mask_np, _ = adz.reset()
                    obs_t = torch.from_numpy(obs_np).to(device)
                    state_t = torch.from_numpy(state_np).to(device)
                    env_mask_t = torch.from_numpy(env_mask_np).to(device)
                    msg_pool_t = torch.zeros((A, D_MSG), device=device)
                    h_t = ppo.policy.init_hidden(A, device)
                    break
            # --- end of k steps
            buf.add_macro_group(
                logp_grp_tau=logp_tau_blk.clone(),
                P_tau=P_tau_block.clone(),
                G=G_block.clone(),
                t_start=t_block_start,
            )

        with torch.no_grad():
            # group-aware critic bootstrap: need last P_tau; use the last block's P_tau
            last_P_tau = (
                block_P_tau[-1]
                if len(block_P_tau) > 0
                else torch.full((A, effective_m_groups), 1.0 / effective_m_groups, device=device)
            )
            last_v = ppo.value(state_t.unsqueeze(0), last_P_tau.unsqueeze(0)).squeeze(0)

        # push cached per-step group features into the buffer first (so ppo.update sees them)
        buf.set_group_tensors(
            grp_P_tau=torch.stack(perstep_grp_P_tau, dim=0),  # (T, A, M)
            grp_G=torch.stack(perstep_grp_G, dim=0),  # (T, A, A)
        )
        # also set last value for GAE
        buf.set_last_value(last_v)

        # finalize (your buffer will now also return grouping-per-block caches)
        batch, aux = buf.finalize()

        # For convenience here:
        block_logp_tau = [t.clone() for t in aux["block_logp_tau"]]
        block_P_tau = [t.clone() for t in aux["block_P_tau"]]
        block_G = [t.clone() for t in aux["block_G"]]
        block_tstarts = [int(t) for t in aux["block_tstarts"]]

        buf.reset()

        # ========== COUNTERFACTUAL ADVANTAGE COMPUTATION ==========
        T = batch["obs"].shape[0]
        A = batch["obs"].shape[1]
        device = batch["state"].device

        if cfg.ablation_no_counterfactual:
            # --- ABLATION: No Counterfactual ---
            # Use GAE advantages (computed by PPO) for comm heads too
            with torch.no_grad():
                V_traj = ppo.value(batch["state"], batch["grp_P_tau"])
                returns, adv_gae = compute_gae(
                    batch["rewards"], V_traj, batch["dones"],
                    cfg.ppo.gamma, cfg.ppo.gae_lambda, batch["last_value"]
                )
                # Normalize advantage
                adv_gae = (adv_gae - adv_gae.mean()) / adv_gae.std().clamp_min(1e-6)
            
            # Use GAE advantage for both send and recv heads
            batch["A_send"] = adv_gae
            batch["A_recv"] = adv_gae
            
            # Empty U_pair_blocks for grouping (no pairwise utility signal)
            U_pair_blocks = [torch.zeros(A, A, device=device) for _ in block_tstarts]
            
            # No comm critic loss to log
            loss_comm_total = torch.tensor(0.0, device=device)

        # --- 6.2 COMM-CRITIC TRAINING (V and Q together) ---
        # Common unpacking (needed for grouping update even in ablation case)
        msg_vis = batch["msg_pool_visible"]
        z_msg_t = batch["z_msg"]
        send_t = batch["send"]
        recv_t = batch["recv"]
        P_tau_t = batch["grp_P_tau"]
        state_t = batch["state"]

        # Skip counterfactual computation if ablation_no_counterfactual (A_send/A_recv already set above)
        if not cfg.ablation_no_counterfactual:
            comm_critic.train()

            # 1) V target and loss
            state_tile = state_t.unsqueeze(1).expand(T, A, state_dim)
            agent_feat = torch.cat(
                [z_msg_t, msg_vis, send_t.float().unsqueeze(-1), P_tau_t], dim=-1
            )

            with torch.no_grad():
                V_traj = ppo.value(state_t, P_tau_t)
                V_next = torch.zeros_like(V_traj)
                V_next[:-1] = V_traj[1:]
                dones_ = (
                    batch["dones"]
                    if batch["dones"].ndim == 2
                    else batch["dones"].unsqueeze(-1).expand_as(V_traj)
                )
                td_target = batch["rewards"] + cfg.ppo.gamma * V_next * (
                    1.0 - dones_.float()
                )

            V_pred = comm_critic.value_withmsg(
                state_tile.reshape(T * A, -1), agent_feat.reshape(T * A, -1)
            ).view(T, A)
            loss_comm_v = F.mse_loss(V_pred, td_target)

            # 2) Build t+1 features once (targets only)
            with torch.no_grad():
                state_tile_tp1 = state_t.unsqueeze(1).expand(T, A, state_t.size(-1))
                agent_feat_tp1 = torch.cat(
                    [z_msg_t, msg_vis, send_t.float().unsqueeze(-1), P_tau_t], dim=-1
                )
                state_tile_tp1 = torch.roll(state_tile_tp1, -1, 0)
                state_tile_tp1[-1].zero_()
                agent_feat_tp1 = torch.roll(agent_feat_tp1, -1, 0)
                agent_feat_tp1[-1].zero_()
                V_with_tp1 = comm_critic_tgt.value_withmsg(
                    state_tile_tp1.reshape(T * A, -1), agent_feat_tp1.reshape(T * A, -1)
                ).view(T, A)

            # 3) Per-sender deltas → A_send + Q supervision
            A_send = torch.zeros(T, A, device=device)
            q_loss_terms = []

            sqrtD = float(z_msg_t.size(-1)) ** 0.5

            for t_idx in range(T - 1):
                # actual senders and their chosen recipients at t
                i_idx = (send_t[t_idx].long() == 1).nonzero(as_tuple=False).squeeze(-1)
                if i_idx.numel() == 0:
                    continue
                j_idx = recv_t[t_idx][i_idx]  # (K,)

                # ---- 1) recompute attention scores at time t (param-free: q=z_j, k=z_i, v=z_i)
                z_send = z_msg_t[t_idx][i_idx]  # (K, D)
                z_recv = z_msg_t[t_idx][j_idx]  # (K, D)
                scores = (z_send * z_recv).sum(dim=-1) / sqrtD  # (K,)
                # ---- per-recipient max for stability
                # compute m_j = max_{i': j} s_{j,i'}
                max_per = torch.full((A,), -1e9, device=device)
                max_per.scatter_reduce_(0, j_idx, scores, reduce="amax")  # PyTorch ≥2.0
                # if you're on older torch: loop by unique j_idx and take max manually

                # shifted exponents and partition Z' = sum_i exp(s - m_j)
                e_shift = torch.exp(scores - max_per[j_idx])  # (K,)
                Z_shift = torch.zeros(A, device=device).scatter_add_(
                    0, j_idx, e_shift
                )  # (A,)

                # ---- exact numerator using the **visible** mailbox at t+1
                msg_vis_tp1 = msg_vis[t_idx + 1]  # (A, D_msg)
                Num_shift = msg_vis_tp1 * Z_shift.unsqueeze(-1)  # (A, D_msg)

                # ---- leave-one-out: μ_j^{(-i)} = (Num - e_shift*v_i) / (Z_shift - e_shift)
                num_ij = Num_shift[j_idx] - e_shift.unsqueeze(-1) * z_send
                den_ij = (Z_shift[j_idx] - e_shift).clamp_min(1e-6).unsqueeze(-1)
                mu_cf = num_ij / den_ij
                # single-sender case ⇒ zero inbox
                mu_cf[(Z_shift[j_idx] - e_shift) <= 1e-6] = 0.0

                # ---- 4) build a batch where ONLY recipient j's mailbox row is swapped to μ_j^{(-i)}
                K = i_idx.numel()
                agent_feat_cf = (
                    agent_feat_tp1[t_idx + 1].unsqueeze(0).expand(K, A, -1).clone()
                )
                zdim = z_msg_t.size(-1)
                msg_slice = slice(zdim, zdim + msg_vis.size(-1))
                agent_feat_cf[torch.arange(K, device=device), j_idx, msg_slice] = mu_cf

                state_cf = state_tile_tp1[t_idx + 1].unsqueeze(0).expand(K, A, -1)

                with torch.no_grad():
                    V_no_all = comm_critic_tgt.value_withmsg(
                        state_cf.reshape(K * A, -1), agent_feat_cf.reshape(K * A, -1)
                    ).view(
                        K, A
                    )  # (K, A)
                V_no_pairs = V_no_all[torch.arange(K, device=device), j_idx]  # (K,)

                # ---- 5) sender-specific deltas
                delta = V_with_tp1[t_idx + 1, j_idx] - V_no_pairs  # (K,)
                # if cfg.comm_penalty_kappa > 0.0:
                #     delta = delta - cfg.comm_penalty_kappa

                A_send[t_idx, i_idx] = delta

                # Q supervision on observed (i,j)
                state_ij = state_t[t_idx + 1].expand(i_idx.numel(), -1)
                send_feat_ij = torch.cat(
                    [
                        z_msg_t[t_idx][i_idx],
                        torch.zeros(i_idx.numel(), msg_vis.size(-1), device=device),
                        torch.ones(i_idx.numel(), 1, device=device),
                        P_tau_t[t_idx][i_idx],
                    ],
                    dim=-1,
                )
                recv_feat_ij = torch.cat(
                    [
                        z_msg_t[t_idx + 1][j_idx],
                        msg_vis[t_idx + 1][j_idx],
                        torch.zeros(i_idx.numel(), 1, device=device),
                        P_tau_t[t_idx + 1][j_idx],
                    ],
                    dim=-1,
                )
                q_pred = comm_critic.q_comm_pair(state_ij, send_feat_ij, recv_feat_ij).view(
                    -1
                )
                q_loss_terms.append(F.mse_loss(q_pred, delta))

            loss_comm_q = (
                torch.stack(q_loss_terms).mean()
                if q_loss_terms
                else torch.tensor(0.0, device=device)
            )

            # 4) One step for comm_critic
            opt_comm.zero_grad(set_to_none=True)
            loss_comm = loss_comm_v + loss_comm_q
            loss_comm.backward()
            nn.utils.clip_grad_norm_(comm_critic.parameters(), cfg.ppo.max_grad_norm)
            opt_comm.step()
            tau = 0.005
            with torch.no_grad():
                for pt, p in zip(comm_critic_tgt.parameters(), comm_critic.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * p.data)

            # --- 6.4 Dense U and A_recv ---
            A_recv = torch.zeros(T, A, device=device)
            A_send_full = (
                A_send.clone()
            )  # you already filled senders (removal-based deltas)

            recv_mask_seq = batch.get("recv_mask", None)  # (T, A, A) bool or None

            U_at_step = {}  # t -> U_centered at step t (detach)
            with torch.no_grad():
                for t in range(T - 1):
                    # ---- Build q(i->j) once at this step (sender at t, recipient at t+1)
                    state_pairs = state_t[t + 1].expand(A * A, -1)

                    send_feat_all = torch.cat(
                        [
                            z_msg_t[t],
                            torch.zeros(
                                A, msg_vis.size(-1), device=device
                            ),  # sender inbox term = 0
                            torch.ones(A, 1, device=device),  # send_bit=1 for utility eval
                            P_tau_t[t],
                        ],
                        dim=-1,
                    )
                    send_feat_tile = (
                        send_feat_all.unsqueeze(1).expand(A, A, -1).reshape(A * A, -1)
                    )

                    recv_feat_tile = torch.cat(
                        [
                            z_msg_t[t + 1].unsqueeze(0).expand(A, A, -1),
                            msg_vis[t + 1].unsqueeze(0).expand(A, A, -1),
                            torch.zeros(A, A, 1, device=device),  # recv-side send_bit=0
                            P_tau_t[t + 1].unsqueeze(0).expand(A, A, -1),
                        ],
                        dim=-1,
                    ).reshape(A * A, -1)

                    q_pairs = comm_critic_tgt.q_comm_pair(
                        state_pairs, send_feat_tile, recv_feat_tile
                    ).view(A, A)
                    mask = None
                    if recv_mask_seq is not None:
                        mask = recv_mask_seq[t].bool()  # (A,A)

                    # zero self-edges regardless
                    q_pairs.fill_diagonal_(0.0)

                    if mask is not None:
                        q_mask = mask & ~torch.eye(A, dtype=torch.bool, device=mask.device)
                        # row-mean over valid columns only
                        row_sum = (q_pairs * q_mask).sum(dim=1, keepdim=True)
                        row_cnt = q_mask.sum(dim=1, keepdim=True).clamp_min(1)
                        row_mean = row_sum / row_cnt
                        U_centered = (q_pairs - row_mean) * q_mask.float()
                    else:
                        row_mean = q_pairs.mean(dim=1, keepdim=True)
                        U_centered = q_pairs - row_mean
                        U_centered.fill_diagonal_(0.0)
                    row_std = (
                        U_centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-3)
                    )
                    U_centered = (U_centered / row_std).clamp_(-5.0, 5.0)
                    U_at_step[t] = U_centered.detach()

                    # ---- A_recv for actual senders at this step
                    senders_t = (send_t[t].long() == 1).nonzero(as_tuple=False).squeeze(-1)
                    if senders_t.numel() > 0:
                        chosen_j = recv_t[t][senders_t]
                        A_recv[t, senders_t] = U_centered[senders_t, chosen_j]

                    # ---- Expected add for non-senders, NEGATED for chosen action=0
                    i0 = (send_t[t] == 0).nonzero(as_tuple=False).squeeze(-1)
                    if i0.numel() > 0:
                        # current recv policy (already soft-masked by groups and hard-masked if provided)
                        logits = ppo.evaluate(
                            batch["obs"][t],
                            batch["hidden"][t],
                            G_soft=batch["grp_G"][t],
                            recv_mask=(
                                recv_mask_seq[t] if recv_mask_seq is not None else None
                            ),
                        )["recv_logits"]
                        if recv_mask_seq is not None:
                            logits = logits.masked_fill(
                                ~recv_mask_seq[t].bool(), float("-inf")
                            )
                        pi_j = torch.softmax(
                            logits, dim=-1
                        )  # (A, A); invalid cols are zero-prob if masked

                        Qm = q_pairs
                        if recv_mask_seq is not None:
                            Qm = Qm.masked_fill(
                                ~recv_mask_seq[t].bool(), 0.0
                            )  # zero-out invalid (i,j)
                        exp_add = (pi_j * Qm).sum(dim=1)  # E_j[q(i->j)]
                        # if cfg.comm_penalty_kappa > 0.0:
                        #     alive_mask = (
                        #         recv_mask_seq[t].any(dim=-1).float()
                        #         if recv_mask_seq is not None
                        #         else torch.ones(A, device=device)
                        #     )
                        #     exp_add = exp_add - cfg.comm_penalty_kappa * alive_mask
                        A_send_full[t, i0] = -exp_add[i0]

            # ---- Per-block U for grouping, one entry per block start (same length as block_tstarts)
            # We must reference a step that has a valid t+1; min(t_b, T-2) guarantees it without padding.
            U_pair_blocks = [U_at_step[min(int(t_b), T - 2)] for t_b in block_tstarts]

            # Normalize & clip across ALL entries now that both branches are filled
            mu = A_send_full.mean()
            std = A_send_full.std().clamp_min(1e-6)
            A_send_full = (A_send_full - mu) / std

            # Attach once
            batch["A_send"] = A_send_full
            batch["A_recv"] = A_recv
        # else: batch["A_send"] and batch["A_recv"] already set in ablation block above

        # ---- Build A_grp_blocks (scalar per block) using group-aware critic
        A_grp_blocks = []
        with torch.no_grad():
            for b, t_b in enumerate(block_tstarts):
                P_b = block_P_tau[b]  # (A, M)
                s_b = state_t[t_b].unsqueeze(0)  # (1, Sd)
                v_groups = ppo.group_critic.net(s_b).squeeze(0)
                V_with = (v_groups.unsqueeze(0) * P_b).sum(dim=-1)

                perm = torch.randperm(v_groups.numel(), device=P_b.device)
                V_cf_i = (v_groups[perm].unsqueeze(0) * P_b).sum(dim=-1)

                A_grp_blocks.append((V_with - V_cf_i))

        # PPO update
        metrics = ppo.update(batch, ablation_no_comm=cfg.ablation_no_comm)
        # print(f"[grp dbg] block_P_tau: {len(block_P_tau)} blocks this iter")
        # ---------------- Grouping + Descriptor update ----------------
        if len(block_P_tau) > 0:
            # print(f"[grp dbg] updating grouping at it {it} with {len(block_P_tau)} blocks")
            # build block-diagonal-like G by stacking per-block (we’ll pass blockwise in a loop for clarity)
            L_grp_total = torch.zeros((), device=device)
            grp_logs_acc = {
                "grp/pg": 0.0,
                "grp/edge": 0.0,
                "grp/balance": 0.0,
                "grp/row_entropy": 0.0,
                "grp/total": 0.0,
            }

            lambda_edge = grp_lambda_edge_sched.at(it)
            lambda_bal = 1.0
            lambda_ent = 0.02

            for b in range(len(block_P_tau)):
                P_b = block_P_tau[b]  # (A, M)
                G_b = block_G[b]  # (A, A)
                logp_b = block_logp_tau[b]  # (A,)
                U_b = (
                    U_pair_blocks[b]
                    if b < len(U_pair_blocks)
                    else torch.zeros(A, A, device=device)
                )
                A_grp_b = A_grp_blocks[b].expand_as(
                    logp_b
                )  # broadcast scalar over agents
                A_grp_b = A_grp_b - A_grp_b.mean().detach()  # center advantage
                A_grp_b = (
                    A_grp_b / (A_grp_b.std().clamp_min(1e-6)).detach()
                )  # normalize advantage
                L_b, logs_b = grouping.loss(
                    logp_grp_tau=logp_b,
                    P_tau=P_b,
                    G=G_b,
                    A_grp=A_grp_b,
                    U_pair=U_b,
                    lambda_edge=lambda_edge,
                    lambda_bal=lambda_bal,
                    lambda_ent=lambda_ent,
                )
                L_grp_total = L_grp_total + L_b
                for k, v in logs_b.items():
                    grp_logs_acc[k] += v

            # optional small L2 on descriptor trunk to keep it tight
            loss_reg_desc = 1e-4 * sum(
                (p.pow(2).mean() for p in descriptor.parameters())
            )

            loss_nce = torch.zeros((), device=device)
            loss_ortho = torch.zeros((), device=device)

            # message content loss (critic-shaped, no grads to comm_critic)
            # Skip this entire section if ablation_no_counterfactual (no comm_critic)
            L_msg_total = torch.zeros((), device=device)
            if not cfg.ablation_no_counterfactual:
                send_seq = batch["send"].long()  # [T, A]
                recv_seq = batch["recv"].long()  # [T, A]
                pairs = (send_seq > 0).nonzero(as_tuple=False)  # [K, 2] -> (t, i)

                if pairs.numel() > 0:
                    valid = pairs[:, 0] < (T - 1)  # only t where t+1 exists
                    pairs = pairs[valid]
                    t_sel = pairs[:, 0]
                    i_sel = pairs[:, 1]
                    # Work timestep-by-timestep to avoid recomputing encodings too much
                    t_unique, inv = t_sel.unique(sorted=False, return_inverse=True)

                    # Freeze critic weights but let grads flow to descriptor (z_msg)
                    for p in comm_critic.parameters():
                        p.requires_grad_(False)

                    for k_idx, t_idx in enumerate(t_unique):
                        mask = inv == k_idx
                        i_t = i_sel[mask]  # sender indices at t
                        j_t = recv_seq[t_idx, i_t]  # chosen recipients at t

                        # Recompute z_msg at t (senders) and t+1 (recipients), with grads
                        obs_core_t = batch["obs"][t_idx, :, : descriptor.cfg.obs_dim]
                        obs_core_tp1 = batch["obs"][t_idx + 1, :, : descriptor.cfg.obs_dim]
                        hid_t = batch["hidden"][t_idx]
                        hid_tp1 = batch["hidden"][t_idx + 1]

                        heads_t = descriptor.heads(
                            {"obs": obs_core_t, "hidden": hid_t},
                            device=device,
                            update_norm=False,
                        )
                        heads_tp1 = descriptor.heads(
                            {"obs": obs_core_tp1, "hidden": hid_tp1},
                            device=device,
                            update_norm=False,
                        )

                        z_send = heads_t["z_msg"][i_t]  # grads to descriptor
                        z_recv = heads_tp1["z_msg"][j_t]  # grads to descriptor

                        # Assemble the same feature templates your comm_critic expects
                        B = i_t.numel()
                        state_tp1 = batch["state"][t_idx + 1].expand(B, -1).detach()

                        zeros_inbox = torch.zeros(
                            B, msg_vis.size(-1), device=device
                        )  # sender inbox = 0 when evaluating utility
                        ones_send = torch.ones(
                            B, 1, device=device
                        )  # send bit = 1 for utility
                        P_send = batch["grp_P_tau"][t_idx][i_t].detach()
                        recv_inbox = batch["msg_pool_visible"][t_idx + 1][j_t].detach()
                        P_recv = batch["grp_P_tau"][t_idx + 1][j_t].detach()

                        send_feat = torch.cat(
                            [z_send, zeros_inbox, ones_send, P_send], dim=-1
                        )  # [B, Df]
                        recv_feat = torch.cat(
                            [z_recv, recv_inbox, torch.zeros(B, 1, device=device), P_recv],
                            dim=-1,
                        )

                        # Forward through critic WITHOUT creating grads on critic params,
                        # but KEEP grads to z_send/z_recv (descriptor params).
                        q_pred = comm_critic.q_comm_pair(
                            state_tp1, send_feat, recv_feat
                        )  # [B]
                        L_msg_total = L_msg_total - q_pred.mean()

                    # Re-enable critic grads for its own optimizer later
                    for p in comm_critic.parameters():
                        p.requires_grad_(True)

            # ---- total descriptor loss (grouping + message + regularizers [+ optional NCE/ortho]) ----
            opt_grouping.zero_grad(set_to_none=True)
            opt_desc.zero_grad(set_to_none=True)

            total_desc_loss = (
                L_grp_total + loss_reg_desc + cfg.msg_loss_coef * L_msg_total
            )
            # If you keep NCE/ortho later:
            # total_desc_loss = L_grp_total + loss_reg_desc + cfg.msg_loss_coef * L_msg_total + lambda_nce * loss_nce + loss_ortho

            total_desc_loss.backward()
            nn.utils.clip_grad_norm_(grouping.parameters(), 1.0)
            nn.utils.clip_grad_norm_(descriptor.parameters(), 1.0)
            opt_grouping.step()
            opt_desc.step()

            # logging
            grp_logs = {
                **{k: v / max(1, len(block_P_tau)) for k, v in grp_logs_acc.items()},
                "comm/nce_loss": loss_nce.detach().item(),
                "comm/ortho_loss": loss_ortho.detach().item(),
            }

        else:
            grp_logs = {
                "grp/pg": 0.0,
                "grp/edge": 0.0,
                "grp/balance": 0.0,
                "grp/row_entropy": 0.0,
                "grp/total": 0.0,
                "comm/nce_loss": 0.0,
                "comm/ortho_loss": 0.0,
            }

        # ---------- Checkpoint every N training iterations ----------
        if it % cfg.ckpt_every_iter_latest == 0:
            ckpt = make_ckpt(
                ppo, grouping, descriptor,
                comm_critic=comm_critic,
                comm_critic_tgt=comm_critic_tgt,
                opt_comm=opt_comm,
                opt_grouping=opt_grouping,
                opt_desc=opt_desc,
                step=global_step,
                train_iter=it,
                cfg=cfg,
                run_id=run_id,
                run_dir=run_dir,
                tb_run_dir=tb_run_dir,
            )
            save_latest(latest_dir, ckpt)

        if it % cfg.ckpt_every_iter_snapshot == 0:
            ckpt = make_ckpt(
                ppo, grouping, descriptor,
                comm_critic=comm_critic,
                comm_critic_tgt=comm_critic_tgt,
                opt_comm=opt_comm,
                opt_grouping=opt_grouping,
                opt_desc=opt_desc,
                step=global_step,
                train_iter=it,
                cfg=cfg,
                run_id=run_id,
                run_dir=run_dir,
                tb_run_dir=tb_run_dir,
            )
            save_snapshot(run_dir, it, ckpt)

        if it % cfg.log_every == 0:
            ret_env_avg = (
                sum(ret_env_window) / len(ret_env_window)
                if ret_env_window
                else float("nan")
            )
            ret_net_avg = (
                sum(ret_net_window) / len(ret_net_window)
                if ret_net_window
                else float("nan")
            )
            len_avg = sum(len_window) / len(len_window) if len_window else float("nan")

            # ----- TensorBoard scalars: PPO -----
            writer.add_scalar("ppo/loss_pi", metrics["loss_pi"], it)
            writer.add_scalar("ppo/loss_v", metrics["loss_v"], it)
            writer.add_scalar("ppo/entropy", metrics["entropy"], it)
            writer.add_scalar("ppo/approx_kl", metrics["approx_kl"], it)
            writer.add_scalar("ppo/clipfrac", metrics["clipfrac"], it)
            writer.add_scalar("ppo/comm_rate", metrics["comm_rate"], it)

            # ----- TensorBoard scalars: returns -----
            writer.add_scalar("returns/avg_env", ret_env_avg, it)
            writer.add_scalar("returns/avg_net", ret_net_avg, it)
            writer.add_scalar("returns/len_avg", len_avg, it)
            writer.add_scalar("returns/completed_eps", float(completed_eps), it)

            # ----- TensorBoard scalars: grouping loss terms -----
            writer.add_scalar("group/total", grp_logs.get("grp/total", 0.0), it)
            writer.add_scalar("group/balance", grp_logs.get("grp/balance", 0.0), it)
            writer.add_scalar(
                "group/row_entropy", grp_logs.get("grp/row_entropy", 0.0), it
            )
            writer.add_scalar("group/pg", grp_logs.get("grp/pg", 0.0), it)
            writer.add_scalar("group/edge", grp_logs.get("grp/edge", 0.0), it)

            writer.add_scalar("comm/nce_loss", grp_logs.get("comm/nce_loss", 0.0), it)
            if cfg.desc.ortho_coef > 0.0:
                writer.add_scalar(
                    "comm/ortho_loss", grp_logs.get("comm/ortho_loss", 0.0), it
                )

            # ----- TensorBoard histogram: group sizes over macro-blocks this iter -----
            if len(group_sizes_all) > 0:
                arr = np.stack(group_sizes_all, axis=0)  # [num_blocks, m_groups]
                writer.add_histogram("group/sizes_hist", arr.flatten(), it)

                # Also log per-group means (nice quick glance)
                means = arr.mean(axis=0)
                writer.add_scalars(
                    "group/size_mean_per_group",
                    {f"g{g}": float(means[g]) for g in range(effective_m_groups)},
                    it,
                )

            # ----- TensorBoard text: sample grouping map (every N iters) -----
            if (it % cfg.log_groups_every == 0) and (sample_group_json is not None):
                writer.add_text(
                    "group/sample_assignment",
                    "```json\n" + json.dumps(sample_group_json, indent=2) + "\n```",
                    it,
                )

            writer.flush()

            # (console print)
            print(
                f"[it {it}] "
                f"pi={metrics['loss_pi']:.3f} v={metrics['loss_v']:.3f} H={metrics['entropy']:.3f} "
                f"H_env={metrics.get('entropy_env', 0.0):.3f} "
                f"H_send={metrics.get('entropy_send', 0.0):.3f} "
                f"H_recv={metrics.get('entropy_recv', 0.0):.3f} | "
                f"KL={metrics['approx_kl']:.4f} clipfrac={metrics['clipfrac']:.3f} "
                f"KL_env={metrics.get('approx_kl_env', 0.0):.4f} clipfrac_env={metrics.get('clipfrac_env', 0.0):.3f} "
                f"KL_send={metrics.get('approx_kl_send', 0.0):.4f} clipfrac_send={metrics.get('clipfrac_send', 0.0):.3f} "
                f"KL_recv={metrics.get('approx_kl_recv', 0.0):.4f} clipfrac_recv={metrics.get('clipfrac_recv', 0.0):.3f} "
                f"comm_rate={metrics['comm_rate']:.3f} | "
                f"send_mask_frac={metrics.get('send_mask_frac', 0.0):.3f} "
                f"grp_total={grp_logs.get('grp/total', 0.0):.3f} "
                f"bal={grp_logs.get('grp/balance', 0.0):.3f} "
                f"rowH={grp_logs.get('grp/row_entropy', 0.0):.3f} "
                f"pg={grp_logs.get('grp/pg', 0.0):.3f} || "
                f"edge={grp_logs.get('grp/edge', 0.0):.3f} | "
                f"NCE={grp_logs.get('comm/nce_loss', 0.0):.3f} "
                f"Ort={grp_logs.get('comm/ortho_loss', 0.0):.3f} | "
                f"eps={completed_eps} "
                f"R_env={ret_env_avg:.2f} R_net={ret_net_avg:.2f} L={len_avg:.1f} "
            )

    ckpt = make_ckpt(
        ppo, grouping, descriptor,
        comm_critic=comm_critic,
        comm_critic_tgt=comm_critic_tgt,
        opt_comm=opt_comm,
        opt_grouping=opt_grouping,
        opt_desc=opt_desc,
        step=global_step,
        train_iter=cfg.iters,
        cfg=cfg,
        run_id=run_id,
        run_dir=run_dir,
        tb_run_dir=tb_run_dir,
    )
    save_snapshot(run_dir, cfg.iters, ckpt)
    save_latest(latest_dir, ckpt)
    writer.flush()
    writer.close()
    # Write final metrics for sensitivity analysis (M, K heatmap)
    ret_env_final = (
        sum(ret_env_window) / len(ret_env_window) if ret_env_window else float("nan")
    )
    final_metrics = {
        "m_groups": cfg.m_groups,
        "k_macro": cfg.k_macro,
        "final_ret_env": ret_env_final,
        "iters": cfg.iters,
        "task": cfg.task,
        "run_dir": str(run_dir),
    }
    with open(run_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("Training complete.")

if __name__ == "__main__":
    cfg = TrainCfg()
    # To switch task:
    # cfg.task = "advpursuit_pretrained"; cfg.map_size = 50
    train(cfg)
