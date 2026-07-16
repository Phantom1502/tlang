from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple
from transformers import TrainingArguments

@dataclass
class DataArguments:
    dataset_name: str
    
    dataset_mode: Literal["on_the_fly", "pre_tokenized"] = "pre_tokenized"
    train_split: str = "train"
    eval_split: str = "val"
    
    max_length: int = 512
    
@dataclass
class HubConfig:
    """Cấu hình huggingface_hub."""
    repo_id : Optional[str] = None
    hf_token: Optional[str] = None