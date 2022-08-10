import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class WeightDecay(torch.nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay <= 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        # Changed this to full backwards hook
        #UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad
        # _input. Please use register_full_backward_hook to get the documented behavior.
        # warnings.warn("Using a non-full backward hook when the forward contains multiple autograd Nodes "
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.
    Example::
        import torchlayers as tl
        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)
    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.
    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").
    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, readout_mask=None, recurrent_mask=None, recurrent_override_values=None, readout_override_values=None):
        """

        :param inputs:
        :param rnn_hxs:
        :param masks:
        :param deterministic:
        :param readout_mask:
        :param recurrent_mask:
        :param recurrent_override_values: e.g. [None, 0.3, 0.4, None, None] where len == n rnn cells
        :return:
        """

        if (readout_mask is not None) and (recurrent_mask is not None):
            RuntimeError('Readout mask and recurrent mask were enabled together. Why would you do this?!')

        if (readout_override_values or recurrent_override_values) and ((readout_mask is not None) or (recurrent_mask is not None)):
            RuntimeError('Cannot use override values with readout/recurrent mask')

        if recurrent_mask is not None:
            assert recurrent_mask.ndim == 1
            assert rnn_hxs.shape[0] == 1
            assert rnn_hxs.shape[1] == recurrent_mask.shape[0]
            rnn_hxs = rnn_hxs * recurrent_mask

        if recurrent_override_values is not None:
            assert rnn_hxs.shape[0] == 1
            assert rnn_hxs.shape[1] == len(recurrent_override_values)

            for i, value in enumerate(recurrent_override_values):
                if value is not None:
                    rnn_hxs[0][i] = value

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        if readout_override_values is not None:
            assert actor_features.shape[0] == 1
            assert actor_features.shape[1] == len(readout_override_values)

            for i, value in enumerate(readout_override_values):
                if value is not None:
                    actor_features[0][i] = value

        if readout_mask is not None:
            assert readout_mask.shape[0] == actor_features.shape[1]
            actor_features = actor_features * readout_mask

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, rnn_cell):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            if rnn_cell == 'gru':
                raise NotImplementedError('GRU not implemented')
            elif rnn_cell == 'rnn':
                print(f'Using Elman RNN')
                self.gru = nn.RNN(recurrent_input_size, hidden_size, nonlinearity='relu')
            elif rnn_cell == 'gru_vanilla':
                print(f'Using vanilla GRU with Tanh nonlinearity')
                self.gru = nn.GRU(recurrent_input_size, hidden_size)
            else:
                raise ValueError(f'Invalid RNN cell {rnn_cell}')

            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False,
                 hidden_size=512,
                 dropout=0,
                 l1=0,
                 recurrent_dropout=0,
                 readout_dropout=0,
                 rnn_cell='rnn'
                 ):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size, rnn_cell=rnn_cell)

        if readout_dropout > 0:
            raise NotImplementedError

        print(f'Using CNN model - not all parameters implemented yet')

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
           init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
           init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
           init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
           init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, dropout=0, l1=0, recurrent_dropout=0, rnn_cell='rnn', readout_dropout=0):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size, rnn_cell=rnn_cell)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.dropout = nn.Dropout(p=dropout)
        self.recurrent_dropout = nn.Dropout(p=recurrent_dropout)
        self.readout_dropout = nn.Dropout(p=readout_dropout)
        if l1 > 0:
            print(f'RNN is now L1 regularised')
            self.gru = L1(self.gru, l1)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.dropout(x)
        rnn_hxs = self.recurrent_dropout(rnn_hxs)

        x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = self.readout_dropout(x)

        hidden_critic = x
        hidden_actor = x

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
