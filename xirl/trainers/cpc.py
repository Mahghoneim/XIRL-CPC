# coding=utf-8
# Copyright 2025 The Google Research Authors.

"""CPC trainer."""

from typing import Dict, List, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class CPCTrainer(Trainer):
    """A trainer for Contrastive Predictive Coding [1].

    References:
        [1]: arxiv.org/abs/1807.03748
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        config,
    ):
        super().__init__(model, optimizer, device, config)

        self.temperature = config.loss.cpc.temperature
        self.prediction_steps = config.loss.cpc.prediction_steps
        self.use_negative_sampling = config.loss.cpc.use_negative_sampling
        
        # Autoregressor to predict future embeddings
        embed_dim = config.model.embedding_size
        hidden_dim = config.loss.cpc.autoregressor_hidden_dim
        
        self.autoregressor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        ).to(device)
        
        # Add autoregressor parameters to optimizer
        self._optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.autoregressor.parameters()),
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
        
        # Queue for negative samples (like MoCo)
        if self.use_negative_sampling:
            self.queue_size = config.loss.cpc.num_negatives
            self.momentum = config.loss.cpc.momentum
            
            # Initialize queue with random normalized vectors
            self.queue = torch.randn(self.queue_size, embed_dim, device=device)
            self.queue = F.normalize(self.queue, dim=1)
            self.queue_ptr = 0
            
            # Key encoder (momentum encoder) - initialized as copy of main model
            self.key_encoder = self._create_key_encoder(model)
            
            # Store new embeddings to update queue after backward pass
            self.new_embeddings_buffer = []

    def _create_key_encoder(self, model):
        """Create momentum encoder as a copy of the main model."""
        # Use deepcopy to create a separate instance
        key_encoder = copy.deepcopy(model)
        
        # Move to device
        key_encoder = key_encoder.to(self._device)
        
        # Set requires_grad=False for momentum encoder
        for param in key_encoder.parameters():
            param.requires_grad = False
            
        return key_encoder

    def compute_loss(
        self,
        embs,
        batch,
    ):
        """Compute CPC loss.
        
        Args:
            embs: Tensor of shape [batch, seq_len, embed_dim] from query encoder
            batch: Dictionary containing batch data
        
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, embed_dim = embs.shape
        
        # Get key embeddings from momentum encoder
        if self.use_negative_sampling:
            with torch.no_grad():  # No gradient for key encoder
                frames = batch["frames"].to(self._device)
                key_embs = self.key_encoder(frames).embs
        else:
            key_embs = embs  # Use same embeddings for simple case
        
        total_loss = 0
        num_predictions = 0
        
        # For each prediction step (1, 2, ..., prediction_steps)
        for k in range(1, self.prediction_steps + 1):
            if seq_len <= k:
                continue
                
            # Current embeddings (z_t) from query encoder
            z_current = embs[:, :-k]  # [batch, seq_len-k, dim]
            
            # Future embeddings (z_{t+k}) from key encoder (for positive samples)
            z_future = key_embs[:, k:]  # [batch, seq_len-k, dim]
            
            # Predict future from current
            z_pred = self.autoregressor(z_current)  # [batch, seq_len-k, dim]
            
            # Normalize embeddings
            z_pred = F.normalize(z_pred, dim=-1)
            z_future = F.normalize(z_future, dim=-1)
            
            # Reshape for contrastive loss
            z_pred = z_pred.reshape(-1, embed_dim)  # [batch*(seq_len-k), dim]
            z_future = z_future.reshape(-1, embed_dim)  # Positives
            
            # Compute contrastive loss
            if self.use_negative_sampling:
                loss = self._info_nce_with_queue(z_pred, z_future)
            else:
                loss = self._info_nce_simple(z_pred, z_future)
            
            total_loss += loss
            num_predictions += 1
        
        if num_predictions == 0:
            return torch.tensor(0.0, device=embs.device)
        
        return total_loss / num_predictions
    
    def _info_nce_simple(self, query, positive):
        """Simple InfoNCE loss using batch negatives."""
        batch_size = query.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(query, positive.t()) / self.temperature
        
        # Positive pairs are diagonal
        labels = torch.arange(batch_size, device=query.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    
    def _info_nce_with_queue(self, query, positive):
        """InfoNCE with negative queue (MoCo-style)."""
        batch_size = query.size(0)
        
        # Positive similarity
        pos_sim = torch.sum(query * positive, dim=1) / self.temperature
        
        # Negative similarities (from queue)
        neg_sim = torch.matmul(query, self.queue.t()) / self.temperature
        
        # Combine
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Labels: 0 for positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Store positive embeddings to update queue AFTER backward pass
        self.new_embeddings_buffer.append(positive.detach())
        
        return loss
    
    def after_backward(self):
        """Update queue and momentum encoder after backward pass."""
        if self.use_negative_sampling and self.new_embeddings_buffer:
            # Update momentum encoder with momentum update
            self._momentum_update_key_encoder()
            
            # Update queue with new embeddings
            for new_embeddings in self.new_embeddings_buffer:
                self._update_queue(new_embeddings)
            self.new_embeddings_buffer.clear()
    
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder."""
        with torch.no_grad():
            for param_q, param_k in zip(self._model.parameters(), 
                                         self.key_encoder.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def _update_queue(self, new_embeddings):
        """Update the negative queue with new embeddings."""
        batch_size = new_embeddings.size(0)
        ptr = self.queue_ptr
        
        with torch.no_grad():
            # Check if we need to wrap around
            if ptr + batch_size > self.queue_size:
                # Need to wrap around
                first_part = self.queue_size - ptr
                second_part = batch_size - first_part
                
                # Update end of queue
                self.queue[ptr:] = new_embeddings[:first_part]
                # Update beginning of queue
                self.queue[:second_part] = new_embeddings[first_part:]
            else:
                # No wrap needed
                self.queue[ptr:ptr + batch_size] = new_embeddings
        
        # Update pointer
        self.queue_ptr = (ptr + batch_size) % self.queue_size
    
    def train_one_iter(self, batch):
        """Override to call after_backward."""
        # Call parent's train_one_iter
        loss_dict = super().train_one_iter(batch)
        
        # Update queue and momentum encoder
        self.after_backward()
        
        return loss_dict
    
    def compute_auxiliary_loss(
        self,
        out,  # pylint: disable=unused-argument
        batch,  # pylint: disable=unused-argument
    ):
        """Compute an auxiliary loss on a single batch."""
        return 0.0