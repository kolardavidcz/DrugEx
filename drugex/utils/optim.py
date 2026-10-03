class ScheduledOptim:
    """Transformer learning rate scheduler wrapper applying inverse square root decay with linear warmup."""

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps=4000):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
        """Update scheduled learning rate and advance the optimizer state."""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Clear parameter gradients in the underlying optimizer."""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """Calculate and update learning rate for the current step across all parameter groups."""
        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
