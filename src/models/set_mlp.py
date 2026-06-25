"""Set-to-query RayMLP: encode a set of observed rays to predict query-ray STEC.

Physics motivation
------------------
A single epoch's GNSS rays form a sparse tomographic sample of the
ionospheric electron-density field.  Instead of mapping a single ray
to another single ray, this model ingests a **set** of nearby observed
rays (each with STEC, geometry, time, constellation) and predicts the
STEC of an arbitrary query ray.  The architecture is permutation-
invariant by construction (attention + mean-pool), so the model sees
neighbouring rays as an unordered point-cloud.

Architecture
------------
::

   Query ray (16d target geometry)
       │
       ▼  QueryEncoder(16→128)
    q_enc (1, 128)
       │
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
   Observed rays × K (23d each)         IPP distances to query (K,)
       │                                      │
       ▼  ObsEncoder(23→128)                 │
    o_enc (K, 128)                            │
       │                                      │
       ├─────────────────────┐                │
       │                     │                │
       ▼                     ▼                ▼
   Cross-Attention       Mean-Pool      Spatial bias
   (Q=q_enc, K=V=o_enc)  → (1, 128)        -d/σ
       │
       ▼
    context (1, 128)
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
   Concat [q_enc | ctx | pool]  (1, 384)
       │
       ▼  OutputMLP(384→128→64→1)
    stec_query (1,)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ──────────────────────────────────────────────────────


class _MLPBlock(nn.Sequential):
    """Linear → BatchNorm → ReLU → Dropout."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


# ── encoders ────────────────────────────────────────────────────────────


class QueryEncoder(nn.Module):
    """Encode the target-ray geometry into a 128-d query vector.

    Input:  (B, 16)  — target geometry columns (see TARGET_GEOM_COLS)
    Output: (B, 128)
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            _MLPBlock(16, 64, dropout),
            _MLPBlock(64, 128, dropout),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 128)


class ObsEncoder(nn.Module):
    """Encode each observed ray into a 128-d feature vector.

    Input:  (B, K, in_features)  — stacked [source_geometry + constellation + source_stec]
    Output: (B, K, 128)
    """

    def __init__(self, in_features: int = 23, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            _MLPBlock(in_features, 64, dropout),
            _MLPBlock(64, 128, dropout),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, D = x.shape
        x = x.reshape(B * K, D)
        x = self.net(x)           # (B*K, 128)
        return x.reshape(B, K, 128)


# ── spatial cross-attention ──────────────────────────────────────────────


class SpatialCrossAttention(nn.Module):
    """Multi-head cross-attention with IPP-distance spatial bias.

    Q  comes from the query ray   → (B, 1, d_head*heads) internally (1, heads, d_head)
    K,V come from observed rays   → (B, K, d_head*heads)
    Spatial bias  ∈ (B, 1, K)  is added to attention logits so nearby rays are
    up-weighted regardless of feature similarity.

    Parameters
    ----------
    d_model : int
        Feature dimension (128).
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout.
    spatial_length_scale_km : float
        σ in  bias_jk = -D_jk / σ.  D_jk is the haversine IPP distance in km.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        spatial_length_scale_km: float = 100.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.spatial_length_scale_km = float(spatial_length_scale_km)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self._sqrt_dh = math.sqrt(self.d_head)

    def forward(
        self,
        q: torch.Tensor,           # (B, d_model)
        kv: torch.Tensor,          # (B, Kvar, d_model)
        mask: torch.Tensor | None, # (B, Kvar)  True = valid
        ipp_dist_km: torch.Tensor, # (B, Kvar)  haversine distance in km
    ) -> torch.Tensor:
        B = kv.shape[0]
        Kvar = kv.shape[1]

        # project
        Q = self.W_q(q).view(B, self.n_heads, 1, self.d_head)       # (B, H, 1, dh)
        K_proj = self.W_k(kv).view(B, Kvar, self.n_heads, self.d_head).permute(0, 2, 1, 3)  # (B, H, Kvar, dh)
        V_proj = self.W_v(kv).view(B, Kvar, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # scaled dot-product + spatial bias
        attn = (Q @ K_proj.transpose(-2, -1)) / self._sqrt_dh             # (B, H, 1, Kvar)
        # clip distances so padded ∞ rows don't poison bias with -∞
        dist_safe = ipp_dist_km.clamp(max=5000.0)
        bias = -(dist_safe.unsqueeze(1) / self.spatial_length_scale_km)   # (B, 1, Kvar)
        attn = attn + bias.unsqueeze(1)                                    # (B, H, 1, Kvar)

        if mask is not None:
            attn_mask = (~mask).unsqueeze(1).unsqueeze(2)            # (B, 1, 1, K)
            # If a row has no valid observations, unmask first entry to avoid NaN
            no_valid = (~mask).all(dim=-1)                          # (B,)
            if no_valid.any():
                attn_mask[no_valid, :, :, 0] = False
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        ctx = (attn_weights @ V_proj).squeeze(2)                          # (B, H, dh)
        ctx = ctx.reshape(B, self.d_model)                           # (B, d_model)
        return self.W_o(ctx)                                         # (B, d_model)


# ── output head ──────────────────────────────────────────────────────────


class OutputHead(nn.Module):
    """MLP that maps [q_enc | ctx | pool] → scalar STEC."""

    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = d_model * 3  # q_enc + ctx + pool
        self.net = nn.Sequential(
            _MLPBlock(in_dim, 128, dropout),
            _MLPBlock(128, 64, dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1)


# ── full set-to-query model ──────────────────────────────────────────────


class SetToQueryModel(nn.Module):
    """Predict query-ray STEC from a set of nearby observed rays.

    Parameters
    ----------
    dropout : float
        Dropout rate for all sub-modules.
    spatial_length_scale_km : float
        σ for the spatial bias  -D/σ added to attention logits.
    """

    def __init__(
        self,
        dropout: float = 0.1,
        spatial_length_scale_km: float = 100.0,
        n_obs_features: int = 23,
    ):
        super().__init__()
        self.query_enc = QueryEncoder(dropout=dropout)
        self.obs_enc = ObsEncoder(in_features=n_obs_features, dropout=dropout)
        self.cross_attn = SpatialCrossAttention(
            d_model=128,
            n_heads=4,
            dropout=dropout,
            spatial_length_scale_km=spatial_length_scale_km,
        )
        self.output_head = OutputHead(d_model=128, dropout=dropout)

    def forward(
        self,
        query_geom: torch.Tensor,      # (B, 16)  target ray geometry
        obs_features: torch.Tensor,    # (B, K, 23)  observed rays
        obs_mask: torch.Tensor,        # (B, K)  True = valid observation
        ipp_dist_km: torch.Tensor,     # (B, K)  haversine IPP distances
    ) -> torch.Tensor:
        """Return predicted query-ray STEC (B, 1)."""
        q_enc = self.query_enc(query_geom)                    # (B, 128)
        o_enc = self.obs_enc(obs_features)                    # (B, K, 128)
        ctx = self.cross_attn(q_enc, o_enc, obs_mask, ipp_dist_km)  # (B, 128)
        pool = (o_enc * obs_mask.unsqueeze(-1).float()).sum(dim=1) / obs_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (B, 128)
        combined = torch.cat([q_enc, ctx, pool], dim=-1)      # (B, 384)
        return self.output_head(combined)                     # (B, 1)
