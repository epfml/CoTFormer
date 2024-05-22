import torch
from torch import nn

from .predictor import DepthPredictor

class StickBreakingACTPredictor(DepthPredictor):
    def __init__(self, config):
        super().__init__(config)
        self._gate = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, 2, bias=False)
        )
        self.threshold = config.halt_threshold

    def post_init(self):
        nn.init.zeros_(self._gate[-1].weight)

    def forward(self, x):
        init_state = {
            "last_depth": 0,
            "log_never_halt": torch.zeros_like(x[..., 0]),
            "act_loss": None,
            "active_mask": torch.ones_like(x[..., 0]) == 1.,
        }
        return self.CANNOT_PREDICT_DEPTH_IN_ADVANCE, init_state

    def gate(self, h):
        logits = self._gate(h)
        return nn.functional.log_softmax(logits, dim=-1)

    def update_halting(self, log_g, log_never_halt):
        log_halt = log_never_halt[..., None] + log_g
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt
    
    def predict_should_continue(self, x, current_depth, halting_state):
        assert current_depth == halting_state["last_depth"] + 1
        log_never_halt = halting_state["log_never_halt"]
        log_g = self.gate(x)
        p, log_never_halt = self.update_halting(log_g, log_never_halt)
        p_never_halt = log_never_halt.exp()
        
        act_loss = (current_depth * p * halting_state["active_mask"]).sum()
        if halting_state["act_loss"] is not None:
            act_loss = halting_state["act_loss"] + act_loss

        
        if current_depth == self.config.n_repeat:
            is_final = torch.ones_like(p_never_halt) == 1.
        else:
            is_final = p_never_halt < (1 - self.threshold)
        
        p_never_halt = p_never_halt.masked_fill(is_final, 0)    


        halting_state = {
            "act_loss": act_loss,
            "log_never_halt": log_never_halt,
            "last_depth": current_depth,
            "active_mask": ~is_final
        }

        return is_final, p_never_halt, halting_state

    def get_loss(self, halting_state):
        p_never_halt = halting_state["log_never_halt"].exp()
        act_loss = (halting_state["act_loss"] + (p_never_halt * (halting_state["last_depth"] + 1)).sum())
        return act_loss
