"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS, NUM_TOKENS_PER_ARM, LEFT_ARM_DIM, RIGHT_ARM_DIM



def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations


class DualArmActionHead(nn.Module):
    """Dual-arm action head with inter-arm attention."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=14, # Total dim (7+7)
        num_task_tokens=512,
        use_pro_version=False, # DualArmBlock only has Pro version for now or we implement base
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Ensure we are in a dual arm setting
        assert action_dim == LEFT_ARM_DIM + RIGHT_ARM_DIM
        
        self.model = DualArmMLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, # This might need adjustment if input is split? 
            # Actually input_dim to MLPResNet is usually hidden_dim. 
            # The input to ActionHead is (B, 128, D).
            # We will create seed x_L and x_R of shape (B, Chunk, D).
            
            hidden_dim=hidden_dim, 
            output_dim_L=LEFT_ARM_DIM,
            output_dim_R=RIGHT_ARM_DIM,
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)
            proprio_features = proprio_projector(proprio)
            proprio_features = proprio_features.unsqueeze(dim=1)
        else:
            proprio_features = None

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        all_actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]
        
        # Split Action Queries (64 Left, 64 Right)
        h_a_L = all_actions_hidden_states[:, :, :NUM_TOKENS_PER_ARM, :]
        h_a_R = all_actions_hidden_states[:, :, NUM_TOKENS_PER_ARM:, :]

        # Create seed inputs x_L and x_R
        # We assume discrete chunks for each arm
        cond_actions_hidden_states = torch.zeros(
            (batch_size, NUM_ACTIONS_CHUNK * 2, self.hidden_dim), # *2 because we split later? Or create separate?
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  
        
        # Let's create separate seeds directly
        x_L = torch.zeros((batch_size, NUM_ACTIONS_CHUNK, self.hidden_dim), device=device, dtype=actions_hidden_states.dtype)
        x_R = torch.zeros((batch_size, NUM_ACTIONS_CHUNK, self.hidden_dim), device=device, dtype=actions_hidden_states.dtype)

        if phase == "Training":
            # Add noise to both
            noise_L = learnable_random_perturbations(NUM_ACTIONS_CHUNK, self.hidden_dim, device, actions_hidden_states.dtype)
            noise_R = learnable_random_perturbations(NUM_ACTIONS_CHUNK, self.hidden_dim, device, actions_hidden_states.dtype)
            x_L = x_L + noise_L
            x_R = x_R + noise_R

        action = self.model(
            x_L, x_R,
            h_a_L=h_a_L,
            h_a_R=h_a_R,
            p=proprio_features,
            h_t=task_hidden_states
            )

        return action


class DualArmMLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim_L, output_dim_R, use_pro_version=False):
        super().__init__()
        # We don't use input_dim here because we create x_L/x_R manually in Head
        
        self.layer_norm_L = nn.LayerNorm(hidden_dim)
        self.layer_norm_R = nn.LayerNorm(hidden_dim)
        
        # No fc1 needed if x is already hidden_dim. 
        # But wait, L1RegressionActionHead creates zeros of shape (B, Chunk*Dim, Hidden)?
        # Ah, the original code: input_dim = input_dim*ACTION_DIM = 4096*7 ??? 
        # No, `input_dim` passed to L1Head is 4096. `input_dim` passed to MLPResNet is `input_dim*ACTION_DIM`?
        # Let's check original L1Head: `input_dim=input_dim*ACTION_DIM`.
        # And `cond_actions_hidden_states` shape is `(B, Chunk*Dim, Hidden)`.
        # Reshaped to `(B, Chunk, Action_Dim * Hidden)`.
        # So `x` dim is `Action_Dim * Hidden`.
        
        # In Dual Arm, let's simplify. Let `x` be `(B, Chunk, Hidden)`.
        # We will expand output capacity by network depth, not width of x.
        # Or we can keep x wide.
        
        # Let's assume x_L and x_R are (B, Chunk, Hidden).
        self.hidden_dim = hidden_dim
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DualArmMLPResNetBlock(dim=hidden_dim))
            
        self.fc2_L = nn.Linear(hidden_dim, output_dim_L)
        self.fc2_R = nn.Linear(hidden_dim, output_dim_R)
        
    def forward(self, x_L, x_R, h_a_L, h_a_R, h_t, p):
        # x_L, x_R: (B, Chunk, Hidden)
        
        # We skip the initial FC1/Norm since x is created clean as hidden_dim
        # Or should we follow original? Original: LayerNorm(x) -> FC1 -> ReLU
        # Since x is zero+noise, FC1 just projects noise.
        
        # For simplicity, we assume x is already in hidden space.
        
        for block in self.blocks:
            # Assuming h_a, h_t are sliced per block like original?
            # The original code does `h_t[:, i+1, :]`. 
            # We need to support that.
            # But `DualArmActionHead` passes full h_t.
            # We need to handle slicing inside loop.
            # However, `DualArmActionHead` logic above passed `h_t` directly.
            # We need `enumerate`.
            pass
            
        for i, block in enumerate(self.blocks):
            # Slice h_t, h_a for layer i
            # Note: Original uses i+1. 
            cur_h_t = h_t[:, i+1, :]
            cur_h_a_L = h_a_L[:, i+1, :]
            cur_h_a_R = h_a_R[:, i+1, :]
            
            x_L, x_R = block(x_L, x_R, cur_h_t, cur_h_a_L, cur_h_a_R, p)
            
        x_L = self.layer_norm_L(x_L)
        x_R = self.layer_norm_R(x_R)
        
        out_L = self.fc2_L(x_L)
        out_R = self.fc2_R(x_R)
        
        # Concatenate: (B, Chunk, Dim_L + Dim_R)
        return torch.cat([out_L, out_R], dim=-1)


class DualArmMLPResNetBlock(nn.Module):
    """Block with Inter-Arm Attention."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Two separate FFNs? Yes.
        self.ffn_L = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.ffn_R = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        
        # Projections for Left Stream
        self.q_L = nn.Linear(dim, dim)
        self.k_L_self = nn.Linear(dim, dim); self.v_L_self = nn.Linear(dim, dim)
        self.k_L_task = nn.Linear(dim, dim); self.v_L_task = nn.Linear(dim, dim)
        self.k_L_adap = nn.Linear(dim, dim); self.v_L_adap = nn.Linear(dim, dim)
        self.k_L_inter = nn.Linear(dim, dim); self.v_L_inter = nn.Linear(dim, dim) # Attention to Right
        self.o_L = nn.Linear(dim, dim)
        
        # Projections for Right Stream
        self.q_R = nn.Linear(dim, dim)
        self.k_R_self = nn.Linear(dim, dim); self.v_R_self = nn.Linear(dim, dim)
        self.k_R_task = nn.Linear(dim, dim); self.v_R_task = nn.Linear(dim, dim)
        self.k_R_adap = nn.Linear(dim, dim); self.v_R_adap = nn.Linear(dim, dim)
        self.k_R_inter = nn.Linear(dim, dim); self.v_R_inter = nn.Linear(dim, dim) # Attention to Left
        self.o_R = nn.Linear(dim, dim)
        
        self.rope = RotaryPositionEmbedding(self.head_dim)
        self.gating = nn.Parameter(torch.zeros(1))

    def forward(self, x_L, x_R, h_t, h_a_L, h_a_R, p=None):
        # x: (B, T, D)
        # h: (B, K, D)
        
        ratio_g = torch.tanh(self.gating)
        
        # Concat Adapter + Proprio
        if p is not None:
            h_a_L = torch.cat([h_a_L, p], dim=1)
            h_a_R = torch.cat([h_a_R, p], dim=1)
            
        B, T, D = x_L.shape
        
        # Helper for multi-head reshape
        def mh(t, L): return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # === Left Stream ===
        q_L = mh(self.q_L(x_L), T)
        k_L_s, v_L_s = mh(self.k_L_self(x_L), T), mh(self.v_L_self(x_L), T)
        k_L_t, v_L_t = mh(self.k_L_task(h_t), h_t.size(1)), mh(self.v_L_task(h_t), h_t.size(1))
        k_L_a, v_L_a = mh(self.k_L_adap(h_a_L), h_a_L.size(1)), mh(self.v_L_adap(h_a_L), h_a_L.size(1))
        k_L_i, v_L_i = mh(self.k_L_inter(x_R), T), mh(self.v_L_inter(x_R), T) # Attend to Right
        
        # RoPE (Simplified application)
        cos, sin = self.rope(T, x_L.device, x_L.dtype)
        q_L, k_L_s = apply_rope(q_L, k_L_s, cos, sin)
        _, k_L_i = apply_rope(k_L_i, k_L_i, cos, sin) # RoPE for inter-arm? Assuming aligned time steps.
        
        # Attention Scores
        scores_L = [
            torch.matmul(q_L, k_L_s.transpose(-2, -1)),
            torch.matmul(q_L, k_L_a.transpose(-2, -1)),
            torch.matmul(q_L, k_L_t.transpose(-2, -1)) * ratio_g,
            torch.matmul(q_L, k_L_i.transpose(-2, -1)) # Inter-arm
        ]
        attn_L = torch.softmax(torch.cat(scores_L, dim=-1) / math.sqrt(self.head_dim), dim=-1)
        
        vals_L = torch.cat([v_L_s, v_L_a, v_L_t, v_L_i], dim=2)
        out_L = torch.matmul(attn_L, vals_L).transpose(1, 2).contiguous().view(B, T, D)
        out_L = self.o_L(out_L)
        
        # === Right Stream ===
        q_R = mh(self.q_R(x_R), T)
        k_R_s, v_R_s = mh(self.k_R_self(x_R), T), mh(self.v_R_self(x_R), T)
        k_R_t, v_R_t = mh(self.k_R_task(h_t), h_t.size(1)), mh(self.v_R_task(h_t), h_t.size(1))
        k_R_a, v_R_a = mh(self.k_R_adap(h_a_R), h_a_R.size(1)), mh(self.v_R_adap(h_a_R), h_a_R.size(1))
        k_R_i, v_R_i = mh(self.k_R_inter(x_L), T), mh(self.v_R_inter(x_L), T) # Attend to Left
        
        # RoPE
        q_R, k_R_s = apply_rope(q_R, k_R_s, cos, sin)
        _, k_R_i = apply_rope(k_R_i, k_R_i, cos, sin)
        
        scores_R = [
            torch.matmul(q_R, k_R_s.transpose(-2, -1)),
            torch.matmul(q_R, k_R_a.transpose(-2, -1)),
            torch.matmul(q_R, k_R_t.transpose(-2, -1)) * ratio_g,
            torch.matmul(q_R, k_R_i.transpose(-2, -1))
        ]
        attn_R = torch.softmax(torch.cat(scores_R, dim=-1) / math.sqrt(self.head_dim), dim=-1)
        
        vals_R = torch.cat([v_R_s, v_R_a, v_R_t, v_R_i], dim=2)
        out_R = torch.matmul(attn_R, vals_R).transpose(1, 2).contiguous().view(B, T, D)
        out_R = self.o_R(out_R)
        
        # Residual + FFN
        new_x_L = self.ffn_L(out_L + x_L)
        new_x_R = self.ffn_R(out_R + x_R)
        
        return new_x_L, new_x_R


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states
            )

        return action
    

class DiscreteActionHead(nn.Module):
    """MLP-based action head that generates discrete action logits via classification."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        n_action_bins=256,
        num_task_tokens=512,
        use_pro_version=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_action_bins = n_action_bins
        
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim * n_action_bins, # Output logits for all bins
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)
            proprio_features = proprio_projector(proprio)
            proprio_features = proprio_features.unsqueeze(dim=1)
        else:
            proprio_features = None

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations)

        # Forward pass through MLPResNet
        # Output shape: (Batch, Chunk, Action_Dim * N_Bins)
        action_logits_flat = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states
            )

        # Reshape to (Batch, Chunk, Action_Dim, N_Bins)
        action_logits = action_logits_flat.reshape(
            batch_size, NUM_ACTIONS_CHUNK, self.action_dim, self.n_action_bins
        )

        return action_logits


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x
