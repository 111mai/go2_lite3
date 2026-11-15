"""CPU-only checks of the first-stage data contracts, without Isaac Gym."""

import unittest

import torch

from .contracts import (
    ACTION_DIM, HISTORY_LENGTH, OBS_DIM, advance_history, assemble_next_state,
    valid_window_mask, validate_action, validate_observation,
)


class ContractTests(unittest.TestCase):
    def test_actor_dimensions(self):
        validate_observation(torch.zeros(3, OBS_DIM))
        validate_action(torch.zeros(3, ACTION_DIM), 3)
        with self.assertRaises(ValueError):
            validate_observation(torch.zeros(3, 46))
        with self.assertRaises(ValueError):
            validate_action(torch.zeros(3, 13), 3)
        with self.assertRaises(ValueError):
            validate_action(torch.zeros(3, ACTION_DIM, dtype=torch.int64), 3)

    def test_history_oldest_first_and_reset_then_append(self):
        history = torch.arange(HISTORY_LENGTH, dtype=torch.float32).view(1, -1, 1)
        history = history.expand(2, -1, OBS_DIM).clone()
        obs = torch.ones(2, OBS_DIM) * 9
        result = advance_history(history, obs, torch.tensor([False, True]))
        torch.testing.assert_close(result[0, :, 0], torch.tensor([1., 2., 3., 4., 9.]))
        torch.testing.assert_close(result[1, :, 0], torch.tensor([0., 0., 0., 0., 9.]))
        torch.testing.assert_close(history[1, :, 0], torch.arange(5, dtype=torch.float32))
        with self.assertRaises(ValueError):
            advance_history(history[:, :-1], obs, torch.tensor([False, True]))

    def test_done_state_uses_only_genuine_terminal_snapshot(self):
        returned = torch.tensor([[2.], [99.], [88.]])
        done = torch.tensor([False, True, True])
        terminal = torch.tensor([[0.], [3.], [123.]])
        available = torch.tensor([False, True, False])
        result, valid = assemble_next_state(returned, done, terminal, available)
        torch.testing.assert_close(result, torch.tensor([[2.], [3.], [0.]]))
        self.assertTrue(torch.equal(valid, torch.tensor([True, True, False])))
        result, valid = assemble_next_state(returned, done)
        torch.testing.assert_close(result, torch.tensor([[2.], [0.], [0.]]))
        self.assertTrue(torch.equal(valid, torch.tensor([True, False, False])))
        with self.assertRaises(ValueError):
            assemble_next_state(returned, done, terminal)
        with self.assertRaises(ValueError):
            assemble_next_state(returned, done, terminal, torch.tensor([True, False, False]))

    def test_windows_do_not_cross_done_or_episode_change(self):
        done = torch.tensor([[False, False, True], [False, True, False],
                             [False, False, False], [False, False, False],
                             [False, False, True]])
        available = torch.tensor([[True, True, True], [True, True, True],
                                  [True, True, True], [True, False, True],
                                  [True, True, False]])
        episode_id = torch.tensor([[3, 3, 3], [4, 4, 4], [5, 6, 6],
                                   [7, 7, 7], [8, 8, 8]])
        expected = torch.tensor([True, False, False, False, False])
        self.assertTrue(torch.equal(valid_window_mask(done, available, episode_id), expected))
        with self.assertRaises(ValueError):
            valid_window_mask(done[:, :0], available[:, :0], episode_id[:, :0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
