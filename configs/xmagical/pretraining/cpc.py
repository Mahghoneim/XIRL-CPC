# coding=utf-8
# Copyright 2025 The Google Research Authors.

"""CPC config - OPTIMIZED VERSION."""

from base_configs.pretrain import get_config as _get_config


def get_config():
    """CPC config with optimized hyperparameters."""

    config = _get_config()

    config.algorithm = "cpc"
    config.optim.train_max_iters = 4_000
    config.data.num_workers = 0  # Keep at 0 for Windows threading fix
    
    # ============================================== #
    # CRITICAL FIXES FOR BETTER CONVERGENCE
    # ============================================== #
    
    # LEARNING RATE - Most important!
    config.optim.lr = 3e-4  # CHANGED from default 1e-5 (30x increase)
    config.optim.weight_decay = 1e-5  # CHANGED from 1e-4 (reduced regularization)
    
    # Frame sampling (same as before)
    config.frame_sampler.strategy = "uniform"
    config.frame_sampler.uniform_sampler.offset = 0
    config.frame_sampler.num_frames_per_sequence = 40
    
    # Model architecture - ENABLE NORMALIZATION
    config.model.model_type = "resnet18_linear"
    config.model.embedding_size = 32
    config.model.normalize_embeddings = True  # CHANGED from False - critical for CPC!
    
    # CPC-specific parameters - OPTIMIZED
    config.loss.cpc.temperature = 0.1  # CHANGED from 0.07 (easier learning)
    config.loss.cpc.prediction_steps = 3  # CHANGED from 5 (more learnable)
    config.loss.cpc.autoregressor_hidden_dim = 256
    config.loss.cpc.use_negative_sampling = True
    config.loss.cpc.num_negatives = 2048  # CHANGED from 4096 (faster, sufficient)
    config.loss.cpc.momentum = 0.999

    return config
