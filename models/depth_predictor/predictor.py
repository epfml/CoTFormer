import torch
from torch import nn

class DepthPredictor(nn.Module):

    CANNOT_PREDICT_DEPTH_IN_ADVANCE = None
    NO_DEPTH_PROB = None
    UNINITIALIZED_STATE = None
    NO_CONTINUE_PROB = None
    NO_LOSS = None

    def __init__(self, config):
        super().__init__()
        self.config = config

    def post_init(self):
        pass

    def predict_should_continue(self, x, current_depth, halting_state):
        should_continue_prob = torch.ones_like(x)
        is_final = torch.zeros_like(x) == 1.
        return is_final, self.NO_CONTINUE_PROB, self.UNINITIALIZED_STATE

    def forward(self, x):
        return self.CANNOT_PREDICT_DEPTH_IN_ADVANCE, self.UNINITIALIZED_STATE

    def get_loss(self, halting_state):
        return self.NO_LOSS


class InAdvanceDepthPredictor(DepthPredictor):

    def _predict_depth(self, x):
        raise NotImplementedError

    def predict_should_continue(self, x, current_depth, halting_state):
        is_final = current_depth == halting_state["token_depths"]
        return is_final, self.NO_CONTINUE_PROB, halting_state

    def forward(self, x):
        token_depths = self._predict_depth(x)
        return token_depths, {"token_depths": token_depths}
