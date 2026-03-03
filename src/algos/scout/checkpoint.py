# Checkpoint save/load and resume path resolution.
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import torch


def _strip_running_norm_keys(sd: Optional[dict]) -> Optional[dict]:
    if sd is None:
        return None
    sd = dict(sd)
    for pfx in ("obs_norm.", "hid_norm.", "msg_norm."):
        for k in [k for k in sd.keys() if k.startswith(pfx)]:
            sd.pop(k, None)
    return sd


def _dump_running_norm(rn) -> Optional[dict]:
    if rn is None or rn.mean is None:
        return None
    count = rn.count
    if isinstance(count, torch.Tensor):
        count = float(count.item())
    else:
        count = float(count)
    return {"count": count, "mean": rn.mean.detach().cpu(), "var": rn.var.detach().cpu()}


def _load_running_norm(rn, blob: Optional[dict], device):
    if rn is None or blob is None:
        return
    if getattr(rn, "count", None) is None:
        rn.count = torch.tensor(blob["count"], device=device, dtype=torch.float32)
    elif isinstance(rn.count, torch.Tensor):
        rn.count.data.fill_(float(blob["count"]))
    else:
        rn.count = float(blob["count"])
    mean = blob["mean"].to(device)
    var = blob["var"].to(device)
    if getattr(rn, "mean", None) is None:
        rn.mean = mean
    else:
        rn.mean.data.copy_(mean)
    if getattr(rn, "var", None) is None:
        rn.var = var
    else:
        rn.var.data.copy_(var)


def resolve_resume_path(cfg) -> Path:
    if getattr(cfg, "resume_ckpt", ""):
        p = Path(cfg.resume_ckpt)
        if p.is_dir():
            p = p / "latest.pt"
        if not p.exists():
            raise FileNotFoundError(f"resume_ckpt not found: {p}")
        return p
    latest = Path(cfg.save_dir) / f"{cfg.task}_latest" / "latest.pt"
    if latest.exists():
        return latest
    base = Path(cfg.save_dir)
    run_dirs = [
        d for d in base.glob(f"{cfg.task}_*")
        if d.is_dir() and not d.name.endswith("_latest")
    ]
    run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    iter_re = re.compile(r"iter_(\d+)\.pt$")
    for d in run_dirs:
        iters = []
        for p in d.glob("iter_*.pt"):
            m = iter_re.search(p.name)
            if m:
                iters.append((int(m.group(1)), p))
        if iters:
            iters.sort(key=lambda x: x[0], reverse=True)
            return iters[0][1]
        final = d / "final.pt"
        if final.exists():
            return final
    raise FileNotFoundError(
        f"No checkpoint found for task={cfg.task}. "
        f"Expected {latest} or {cfg.task}_*/iter_*.pt under {cfg.save_dir}."
    )


def make_ckpt(
    ppo,
    grouping,
    descriptor,
    comm_critic=None,
    comm_critic_tgt=None,
    opt_comm=None,
    opt_grouping=None,
    opt_desc=None,
    *,
    step: int,
    train_iter: int,
    cfg,
    run_id: str,
    run_dir: Path,
    tb_run_dir: Path,
) -> dict:
    norms = {
        "obs_norm": _dump_running_norm(getattr(descriptor, "obs_norm", None)),
        "hid_norm": _dump_running_norm(getattr(descriptor, "hid_norm", None)),
        "msg_norm": _dump_running_norm(getattr(descriptor, "msg_norm", None)),
    }
    ckpt = {
        "step": step,
        "train_iter": train_iter,
        "cfg": vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg),
        "policy_state": ppo.policy.state_dict(),
        "critic_state": ppo.group_critic.state_dict(),
        "ppo_opt_state": ppo.optimizer.state_dict(),
        "group_state": None if grouping is None else grouping.state_dict(),
        "desc_state": None if descriptor is None else _strip_running_norm_keys(descriptor.state_dict()),
        "group_opt_state": None if opt_grouping is None else opt_grouping.state_dict(),
        "desc_opt_state": None if opt_desc is None else opt_desc.state_dict(),
        "comm_critic_state": None if comm_critic is None else comm_critic.state_dict(),
        "comm_critic_tgt_state": None if comm_critic_tgt is None else comm_critic_tgt.state_dict(),
        "comm_opt_state": None if opt_comm is None else opt_comm.state_dict(),
        "desc_norms": norms,
        "meta": {"run_id": run_id, "run_dir": str(run_dir), "tb_run_dir": str(tb_run_dir)},
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    return ckpt


def save_ckpt(save_dir: Path, name: str, ckpt: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.pt"
    torch.save(ckpt, path)
    print(f"[ckpt] saved -> {path}")
    torch.save(ckpt, save_dir / "latest.pt")


def save_latest(latest_dir: Path, ckpt: dict):
    latest_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, latest_dir / "latest.pt")
    print(f"[ckpt] saved -> {latest_dir / 'latest.pt'}")


def save_snapshot(run_dir: Path, it: int, ckpt: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"iter_{it}.pt"
    torch.save(ckpt, path)
    print(f"[ckpt] saved -> {path}")


def load_ckpt(
    path: Path,
    ppo,
    grouping,
    descriptor,
    device,
    comm_critic=None,
    comm_critic_tgt=None,
    opt_comm=None,
    opt_grouping=None,
    opt_desc=None,
):
    ckpt = torch.load(path, map_location=device)
    ppo.policy.load_state_dict(ckpt["policy_state"])
    ppo.group_critic.load_state_dict(ckpt["critic_state"])
    ppo.optimizer.load_state_dict(ckpt["ppo_opt_state"])
    if grouping is not None and ckpt.get("group_state") is not None:
        grouping.load_state_dict(ckpt["group_state"])
    if descriptor is not None and ckpt.get("desc_state") is not None:
        desc_sd = _strip_running_norm_keys(ckpt["desc_state"])
        res = descriptor.load_state_dict(desc_sd, strict=False)
        allowed_prefixes = ("obs_norm.", "hid_norm.", "msg_norm.")
        bad_missing = [k for k in res.missing_keys if not k.startswith(allowed_prefixes)]
        bad_unexp = [k for k in res.unexpected_keys if not k.startswith(allowed_prefixes)]
        if bad_missing or bad_unexp:
            raise RuntimeError(
                f"Descriptor state_dict mismatch.\nMissing: {bad_missing}\nUnexpected: {bad_unexp}"
            )
        norms = ckpt.get("desc_norms", {})
        _load_running_norm(getattr(descriptor, "obs_norm", None), norms.get("obs_norm"), device)
        _load_running_norm(getattr(descriptor, "hid_norm", None), norms.get("hid_norm"), device)
        _load_running_norm(getattr(descriptor, "msg_norm", None), norms.get("msg_norm"), device)
    if opt_grouping is not None and ckpt.get("group_opt_state") is not None:
        opt_grouping.load_state_dict(ckpt["group_opt_state"])
    if opt_desc is not None and ckpt.get("desc_opt_state") is not None:
        opt_desc.load_state_dict(ckpt["desc_opt_state"])
    if comm_critic is not None and ckpt.get("comm_critic_state") is not None:
        comm_critic.load_state_dict(ckpt["comm_critic_state"])
    if comm_critic_tgt is not None and ckpt.get("comm_critic_tgt_state") is not None:
        comm_critic_tgt.load_state_dict(ckpt["comm_critic_tgt_state"])
        comm_critic_tgt.eval()
        for p in comm_critic_tgt.parameters():
            p.requires_grad_(False)
    if opt_comm is not None and ckpt.get("comm_opt_state") is not None:
        opt_comm.load_state_dict(ckpt["comm_opt_state"])
    if "rng" in ckpt and ckpt["rng"].get("torch", None) is not None:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"].get("torch_cuda", None) is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
    meta = ckpt.get("meta", {})
    return int(ckpt.get("step", 0)), int(ckpt.get("train_iter", 0)), meta
