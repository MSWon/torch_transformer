import numpy as np
import torch

from torch import nn


class WarmupOptimizer(object):
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        self.param_groups = self._optimizer.param_groups
        self.synchronize = self._optimizer.synchronize
        self.skip_synchronize = self._optimizer.skip_synchronize

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        self.lr = self.init_lr * self._get_lr_scale()

        for param_group in self.param_groups:
            param_group['lr'] = self.lr


class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.0, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.is_ignore_index = ignore_index != None
        self.epsilon = epsilon
        self.ignore_index = ignore_index

        assert isinstance(epsilon, float), f"epsilon type must be 'float' yours: {type(epsilon)}"
        if self.is_ignore_index:
            assert isinstance(ignore_index, int), f"ignore_index type must be 'int' yours: {type(ignore_index)}"

    def forward(self, logits, target):
        """
        logits: (batch_size, num_vocabs, sequence length)
        target: (batch_size, sequence length)
        """
        n_class = logits.size(1)

        confidence = torch.as_tensor(1 - self.epsilon)
        low_confidence = torch.as_tensor((1 - confidence) / (n_class - 1))
        
        # (batch_size * sequence length, num_vocabs)
        flatten_logits = logits.permute(0,2,1).reshape(-1, n_class)
        # (batch_size * sequence length,)
        flatten_target = target.reshape(-1)
        
        # (batch_size * sequence length, num_vocabs)
        one_hot = torch.zeros_like(flatten_logits).scatter(1, flatten_target.unsqueeze(-1), 1)
        one_hot = one_hot * confidence + (1 - one_hot) * low_confidence
        # (batch_size * sequence length, num_vocabs)
        neg_logprobs = -nn.functional.log_softmax(flatten_logits, dim=1)
        
        assert one_hot.shape == neg_logprobs.shape
        
        normalizing = -(confidence * torch.log(confidence) + 
                        (n_class - 1) * low_confidence * torch.log(low_confidence + 1e-20))
        
        # (batch_size * sequence length,)
        nll_loss = (one_hot * neg_logprobs).sum(dim=1)
        nll_loss = nll_loss - normalizing
        
        # (batch_size * sequence length,)
        mask = flatten_target.ne(self.ignore_index)
        
        assert nll_loss.shape == mask.shape
        
        masked_loss = nll_loss * mask
        reduced_loss = masked_loss.sum() / mask.sum()
        return reduced_loss