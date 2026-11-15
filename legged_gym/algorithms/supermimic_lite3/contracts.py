"""Validated tensor contracts for a future, isolated Lite3 world-model collector.

These functions do not collect simulator states or implement a training method.
They intentionally require a genuine pre-reset terminal state to accept a done
transition. The current environment's returned observation is not such a state.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


OBS_DIM = 45
HISTORY_LENGTH = 5
ACTION_DIM = 12


def _check_bool_vector(value: Tensor, batch: int, name: str) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ValueError("{} must be a bool tensor".format(name))
    if tuple(value.shape) != (batch,):
        raise ValueError("{} must have shape ({},)".format(name, batch))


def validate_observation(obs: Tensor) -> None:
    """Check a batch of existing Go2Robot's 45-value actor observations."""
    if not isinstance(obs, Tensor) or obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError("observation must have shape (batch, 45)")
    if not obs.is_floating_point():
        raise ValueError("observation must be floating point")


def validate_action(action: Tensor, batch: int) -> None:
    """Check policy actions before applying any environment clipping or scaling."""
    if not isinstance(action, Tensor) or tuple(action.shape) != (batch, ACTION_DIM):
        raise ValueError("action must have shape (batch, 12)")
    if not action.is_floating_point():
        raise ValueError("action must be floating point")


def advance_history(history: Tensor, obs: Tensor, done: Tensor) -> Tensor:
    """Append observations oldest-to-newest, clearing histories on episode reset.

    Mirrors OnPolicyRunnerCTS: after an environment step, reset histories whose
    episode ended, then append that environment's returned observation.
    """
    validate_observation(obs)
    batch = obs.shape[0]
    if not isinstance(history, Tensor) or tuple(history.shape) != (batch, HISTORY_LENGTH, OBS_DIM):
        raise ValueError("history must have shape (batch, 5, 45)")
    if history.dtype != obs.dtype or history.device != obs.device:
        raise ValueError("history and observation must have matching dtype and device")
    _check_bool_vector(done, batch, "done")
    if done.device != obs.device:
        raise ValueError("done and observation must be on the same device")
    cleared = torch.where(done[:, None, None], torch.zeros_like(history), history)
    return torch.cat((cleared[:, 1:, :], obs[:, None, :]), dim=1)


def assemble_next_state(
    returned_state: Tensor,
    done: Tensor,
    terminal_state: Optional[Tensor] = None,
    terminal_available: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Choose true successor states without confusing them with reset states.

    ``returned_state`` is the state returned by a hypothetical collector after
    stepping; its values for done environments MUST be treated as reset data.
    ``terminal_state`` must instead be captured before reset, and availability
    must be explicit. Missing terminal rows are zero-filled and marked invalid;
    consumers must use the returned mask for every loss and sampled window.
    This helper cannot itself verify where a caller obtained a terminal snapshot.
    """
    if not isinstance(returned_state, Tensor) or returned_state.ndim < 2:
        raise ValueError("returned_state must have shape (batch, ...)")
    batch = returned_state.shape[0]
    _check_bool_vector(done, batch, "done")
    if done.device != returned_state.device:
        raise ValueError("done and state must be on the same device")
    if terminal_available is None:
        if terminal_state is not None:
            raise ValueError("terminal availability must be explicit")
        terminal_available = torch.zeros_like(done)
    _check_bool_vector(terminal_available, batch, "terminal_available")
    if terminal_available.device != returned_state.device:
        raise ValueError("terminal availability and state must be on the same device")
    if torch.any(terminal_available & ~done).item():
        raise ValueError("terminal snapshots may only be marked available for done rows")
    if terminal_state is None:
        if torch.any(terminal_available).item():
            raise ValueError("available terminal rows require a terminal_state")
        terminal_state = torch.zeros_like(returned_state)
    elif (not isinstance(terminal_state, Tensor)
          or terminal_state.shape != returned_state.shape
          or terminal_state.dtype != returned_state.dtype
          or terminal_state.device != returned_state.device):
        raise ValueError("terminal_state must match returned_state shape, dtype and device")
    selector = done.reshape((batch,) + (1,) * (returned_state.ndim - 1))
    valid = ~done | terminal_available
    next_state = torch.where(selector, terminal_state, returned_state)
    valid_selector = valid.reshape((batch,) + (1,) * (returned_state.ndim - 1))
    return torch.where(valid_selector, next_state, torch.zeros_like(next_state)), valid


def valid_window_mask(done: Tensor, next_state_available: Tensor, episode_id: Tensor) -> Tensor:
    """Return one validity flag per [batch, time] transition window.

    Every transition needs a real next state. A terminal transition may occur
    only last; a change in the current-state episode ID is never allowed inside
    a window. Timeouts count as done for boundaries even when a later value
    function chooses to bootstrap through that timeout.
    """
    if not isinstance(done, Tensor) or done.dtype != torch.bool or done.ndim != 2:
        raise ValueError("done must be a bool tensor with shape (batch, time)")
    if done.shape[1] == 0:
        raise ValueError("windows must contain at least one transition")
    if (not isinstance(next_state_available, Tensor)
            or next_state_available.dtype != torch.bool
            or next_state_available.shape != done.shape):
        raise ValueError("next_state_available must match done's shape and be bool")
    if (not isinstance(episode_id, Tensor) or episode_id.shape != done.shape
            or episode_id.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("episode_id must match done's shape and be integral")
    if next_state_available.device != done.device or episode_id.device != done.device:
        raise ValueError("window tensors must be on the same device")
    return (next_state_available.all(dim=1)
            & ~done[:, :-1].any(dim=1)
            & (episode_id[:, 1:] == episode_id[:, :-1]).all(dim=1))
