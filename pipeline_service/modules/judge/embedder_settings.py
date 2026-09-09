from __future__ import annotations

from pydantic import BaseModel


class EmbedderConfig(BaseModel):
    """DINOv3 embedder config for the judge best-view stage (S2BV)."""

    enabled: bool = True
    model_id: str = "cont1037/dinov3-vits16-pretrain-lvd1689m"
    revision: str = "ba4586919549aa692fd9b39ab3e9777c564b497d"
    hf_token: str | None = None
    device: str | None = None  
    batch_size: int = 8
    trust_remote_code: bool = False
