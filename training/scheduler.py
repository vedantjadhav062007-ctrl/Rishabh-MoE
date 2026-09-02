import math

class WSDSchedule:
    """
    Warmup-Stable-Decay Learning Rate Schedule.
    Used by DeepSeek models for robust RL training.
    """
    def __init__(self, warmup_steps=500, stable_fraction=0.8, total_steps=5000, lr_min_ratio=0.1):
        self.warmup_steps = warmup_steps
        self.stable_fraction = stable_fraction
        self.total_steps = total_steps
        self.lr_min_ratio = lr_min_ratio
        
        self.decay_start = int(self.total_steps * self.stable_fraction)

    def get_lr_scale(self, step: int) -> float:
        """Returns a multiplier (0.0 to 1.0) for the base learning rate."""
        if step < self.warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.warmup_steps))
            
        elif step < self.decay_start:
            # Stable flat plateau
            return 1.0
            
        else:
            # Cosine decay down to lr_min_ratio
            progress = (step - self.decay_start) / max(1, (self.total_steps - self.decay_start))
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lr_min_ratio + (1.0 - self.lr_min_ratio) * cosine_decay
