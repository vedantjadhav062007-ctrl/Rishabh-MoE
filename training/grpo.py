import torch
import torch.nn.functional as F
import re

def compute_reward(prompt, response, expected_answer):
    """
    Reward function enforcing math correctness, language constraint, and format.
    Prevents language mixing by heavily penalizing non-compliant formats.
    """
    r = 0.0
    
    # 1. Format Compliance (+0.2)
    # Must use <thinking> and <answer> tags
    if re.search(r'<thinking>.*?</thinking>\s*<answer>.*?</answer>', response, re.DOTALL):
        r += 0.2
        
    # 2. Language Constraint (+0.1)
    # Penalize non-ASCII characters (e.g., Chinese characters mixed in)
    if re.fullmatch(r'[\x00-\x7F\s]*', response):
        r += 0.1
        
    # 3. Math Correctness (+1.0)
    # Extract predicted answer
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    predicted = match.group(1).strip() if match else ""
    
    if predicted == expected_answer:
        r += 1.0
    
    # 4. Step-by-Step Reasoning Bonus (+0.1)
    if "Step 1:" in response or "Therefore" in response:
        r += 0.1
        
    # Language mixing cap
    # If they didn't get the language constraint, cap total reward severely
    if r > 0.3 and not re.fullmatch(r'[\x00-\x7F\s]*', response):
        r = 0.3
        
    return min(r, 1.4)

class GRPOTrainer:
    """
    Group Relative Policy Optimization with MTP-aware loss.
    """
    def __init__(self, model, ref_model, offload_engine, config):
        self.model = model
        self.ref_model = ref_model # Kept in CPU memory, swapped in as needed
        self.offload = offload_engine
        self.config = config
        self.beta = config.grpo_kl_beta

    def compute_grpo_loss(self, policy_logits, ref_logits, actions, advantages, mtp_logits_list=None):
        """Computes GRPO loss with KL penalty + MTP auxiliary loss."""
        
        # 1. Standard Policy Loss
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        # Gather logprobs for taken actions
        action_policy_logprobs = torch.gather(policy_logprobs, -1, actions.unsqueeze(-1)).squeeze(-1)
        action_ref_logprobs = torch.gather(ref_logprobs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        # KL Divergence Penalty
        kl = torch.exp(action_ref_logprobs - action_policy_logprobs) - (action_ref_logprobs - action_policy_logprobs) - 1
        
        # GRPO Objective: Advantage * logprob - KL penalty
        loss_policy = -(advantages.unsqueeze(1) * action_policy_logprobs - self.beta * kl).mean()
        
        # 2. MTP (Multi-Token Prediction) Auxiliary Loss
        loss_mtp = 0.0
        if mtp_logits_list is not None and len(mtp_logits_list) > 0:
            for i, mtp_logits in enumerate(mtp_logits_list):
                # Target is shifted i+1 steps into the future
                if actions.shape[1] > i + 1:
                    target_actions = actions[:, i+1:]
                    pred_logits = mtp_logits[:, :-(i+1), :]
                    
                    mtp_logprobs = F.log_softmax(pred_logits, dim=-1)
                    action_mtp_logprobs = torch.gather(mtp_logprobs, -1, target_actions.unsqueeze(-1)).squeeze(-1)
                    
                    # Weight MTP loss by config and advantage
                    loss_mtp -= (advantages.unsqueeze(1)[:, :-(i+1)] * action_mtp_logprobs).mean() * self.config.mtp_loss_weight

        return loss_policy + loss_mtp

    def step(self, prompts, expected_answers, lr_scale):
        """
        Executes one GRPO step:
        1. Generate G rollouts
        2. Compute rewards and relative advantages
        3. Backward pass with CPU offloading
        """
        # (Pseudocode for rollout generation)
        # rollouts, actions = generate_rollouts(self.model, prompts, G=self.config.grpo_rollouts)
        
        # (Pseudocode for rewards)
        # rewards = [compute_reward(p, r, ans) for p, r, ans in zip(prompts, rollouts, expected_answers)]
        
        # Advantages: A_i = (R_i - mean(R)) / (std(R) + 1e-8)
        # ...
        
        # loss = self.compute_grpo_loss(...)
        
        # Custom backward loop with layer-by-layer offload
        # loss.backward(retain_graph=True)
        # for i in reversed(range(self.config.n_layers)):
        #     self.offload.offload_layer_gradients(i)
        
        # self.offload.sync_transfers()
        # self.offload.update_and_stream_weights_to_gpu(lr_scale=lr_scale)
        
        # # Update MoE Load Balancing Bias (Aux-Loss-Free)
        # for layer in self.model.layers:
        #     layer.moe.update_expert_bias()
        
        pass # Full rollout loop omitted for brevity in this shell file
