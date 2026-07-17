from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

@dataclass
class DataArguments:
    dataset_name: str
    
    dataset_mode: Literal["on_the_fly", "pre_tokenized"] = "pre_tokenized"
    train_split: str = "train"
    eval_split: str = "val"
    
    max_length: int = 512
    
    cache_dir: Optional[str] = None