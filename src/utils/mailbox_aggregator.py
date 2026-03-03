import torch
import torch.nn as nn


def segmented_softmax(
    scores: torch.Tensor, seg_ids: torch.Tensor, n_segs: int
) -> torch.Tensor:
    """
    scores: (K,)     unnormalized scores per edge
    seg_ids: (K,)    segment id in [0, n_segs) for each edge (recipient index here)
    returns: (K,)    softmax weights within each segment
    """
    device = scores.device
    # max per segment for stability
    max_per = torch.full((n_segs,), -1e9, device=device)
    max_per.scatter_reduce_(0, seg_ids, scores, reduce="amax")
    exp = torch.exp(scores - max_per[seg_ids])
    denom = torch.zeros(n_segs, device=device).scatter_add_(0, seg_ids, exp)
    return exp / (denom[seg_ids] + 1e-6)


class MailboxAttention(nn.Module):
    """
    Param-free dot-product attention in z-space.
    q_j = z_j, k_i = z_i, v_i = z_i
    pooled_j = sum_i softmax_j( q_j·k_i / sqrt(D) ) * v_i
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**0.5

    @torch.no_grad()
    def forward(
        self,
        z_msg_all: torch.Tensor,  # (A, D)
        recv_query_all: torch.Tensor,  # (A, D) (usually same tensor; kept for API compat)
        i_idx: torch.Tensor,  # (K,) sender indices
        j_idx: torch.Tensor,  # (K,) recipient indices
        A: int,
    ) -> torch.Tensor:  # -> (A, D) pooled messages
        if i_idx.numel() == 0:
            return torch.zeros_like(z_msg_all)

        # Dot-product scores in z-space
        k = z_msg_all[i_idx]  # (K, D)
        q = z_msg_all[j_idx]  # (K, D)  (you may also use recv_query_all[j_idx])
        scores = (q * k).sum(dim=-1) / self.scale  # (K,)

        # Segment-normalize over recipients
        alpha = segmented_softmax(scores, j_idx, A)  # (K,)

        # Weighted sum per recipient
        pooled = torch.zeros_like(z_msg_all)  # (A, D)
        pooled.index_add_(0, j_idx, alpha.unsqueeze(-1) * k)
        return pooled
