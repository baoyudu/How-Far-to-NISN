from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

@dataclass
class TrainingConfig:
    data_path: str
    paradigm: str = 'para02'
    subject: str = 'sub02'
    categories: Optional[List[int]] = field(default_factory=lambda: [7, 38])
    lang: str = 'zh'
    emb_type: str = 'jina'
    encoder_version: int = 1
    embedding_dim: Optional[int] = None
    device: str = 'cuda'
    eegcnn_dropout: float = 0.1
    projection_dropout: float = 0.1
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    temperature: float = 0.07
    margin: float = 0.2
    repeat_range: Tuple[int] = (3, 12)
    ratio_range: Tuple[float] = (0.2, 0.8)
    repeat_step: int = 3
    ratio_step: float = 0.2
    random_seed: Optional[int] = None
    pad_sequences: bool = True
    max_seq_len: Optional[int] = 2921
    seed: int = 42
    gradient_clip: float = 1.0
    patience: int = 5
    min_delta: float = 1e-4
    log_every_n_batches: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    n_samples_to_plot: int = 100
    debug_output: bool = False
    debug_model_structure: bool = False
    debug_embedding_norms: bool = False
    debug_embedding_norm_freq: int = 100
    debug_data_loading: bool = False
    debug_memory_usage: bool = False
    debug_gradient_flow: bool = False
    note: str = ''

    def __post_init__(self):
        if self.embedding_dim is None:
            self.embedding_dim = 1024 if self.emb_type == 'jina' else 768
            
        if self.encoder_version == 2 and not self.pad_sequences:
            print("Warning: Forcing pad_sequences=True for encoder version 2 (FC projection head)")
            self.pad_sequences = True
            
        if not self.debug_output:
            self.debug_model_structure = False
            self.debug_embedding_norms = False
            self.debug_data_loading = False
            self.debug_memory_usage = False
            self.debug_gradient_flow = False 