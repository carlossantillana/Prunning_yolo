import torch
import numpy as np
from typing import Union
#change PERCENTILE to prune at different percentiles.
PERCENTILE = 27
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result

def pre_prune_weights(self):
    # get weights in dict {name: torch.Tensor}
    state_dict = self.net.state_dict()
    # ================================================================ #
    # YOUR CODE HERE:
    #   1.find prunable variables i.e. kernel weight/bias
    #   2.prune parameters based on your threshold, calculated based on input argument percentage
    #   example pseudo code for step 2-3:
    #       for name, var in enumerate(state_dict):
    #           # construct pruning mask
    #           mask = var < threshold
    #           new_var = var[var < threshold]
    #           state_dict[name] = new_var
    # ================================================================ #
    for name, var in state_dict.items():
        if "weight" in str(name) or "bias" in str(name):
            threshold = percentile(torch.abs(var), PERCENTILE)
            state_dict[name] = torch.where(torch.abs(var) > threshold, var, torch.zeros(var.shape).cuda())
    self.net.load_state_dict(state_dict)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

def prune_weights_in_training(self):
# get weights in dict {name: torch.Tensor}
    state_dict = self.net.state_dict()
    # ================================================================ #
    # YOUR CODE HERE:
    #   you can reuse code for pre_prune_weights here
    #       -> make sure pruned weights not recovered
    #   or reselect threshold dynamically
    #       -> make sure pruned percentage same
    # ================================================================ #
    for name, var in state_dict.items():
        if "weight" in str(name) or "bias" in str(name):
            threshold = percentile(torch.abs(var), PERCENTILE)
            state_dict[name] = torch.where(torch.abs(var) > threshold, var, torch.zeros(var.shape).cuda())
    self.net.load_state_dict(state_dict)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
