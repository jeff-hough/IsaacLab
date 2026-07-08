# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Output-equivalence test for the LSTM neural actuator (no Omniverse Kit app).

Compares the Lab actuator (``isaaclab.actuators.ActuatorNetLSTM``) against the
Newton controller (``newton.actuators.ControllerNeuralLSTM``) on the *same*
scripted network, driving each one directly with hard-coded joint states and
comparing the raw (pre-clamp) network effort -- Lab ``computed_effort`` vs the
Newton controller's written ``forces``. Comparing before any DC-motor clamping
means a clamping difference cannot mask a network-input mismatch.

The actuator objects run on plain torch + warp, so no Omniverse Kit app is
required and the test is fast.

Two groups of cases:

* **Zero velocity** -- isolates the position path (``pos_error`` construction,
  the LSTM forward from a zero hidden state, and output scaling). The velocity
  channel is zero on both sides here, so it does not participate.

* **Non-zero velocity** -- exercises the velocity channel, where the two
  implementations build the LSTM input differently: Lab feeds the raw joint
  velocity, while Newton feeds a velocity error (``target_vel - vel``). With a
  zero velocity target (position control), Lab feeds ``+vel`` and Newton feeds
  ``-vel``, so these cases surface any divergence in the velocity channel.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch
import warp as wp

from isaaclab.actuators import ActuatorNetLSTM
from isaaclab.actuators.actuator_net_cfg import ActuatorNetLSTMCfg
from isaaclab.utils.types import ArticulationActions
from newton.actuators import ControllerNeuralLSTM

# Number of joints under test.
_NUM_JOINTS = 3
_ZERO_VEL = [0.0] * _NUM_JOINTS

# Hard-coded (joint_pos, target_pos) pairs evaluated at zero velocity. Includes a
# zero-error case and a spread of positive / negative / mixed position errors.
_ZERO_VELOCITY_CASES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.1, -0.2, 0.3], [0.0, 0.0, 0.0]),
    ([0.5, 0.5, -0.5], [0.0, 0.0, 0.0]),
    ([-0.4, 0.2, 0.7], [0.1, -0.1, 0.2]),
    ([1.2, -1.0, 0.0], [-0.3, 0.4, -0.6]),
]

# Hard-coded (joint_pos, target_pos, joint_vel) triples, evaluated with a zero
# velocity target (position control). Covers the velocity channel.
_VELOCITY_CASES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, -0.1, 0.2]),
    ([0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [0.5, 0.5, -0.5]),
    ([0.5, 0.5, -0.5], [0.1, -0.1, 0.2], [-0.3, 0.7, 1.0]),
    ([-0.4, 0.2, 0.7], [-0.3, 0.4, -0.6], [1.5, -2.0, 0.5]),
]


class _DummyLSTM(torch.nn.Module):
    """Minimal LSTM network for actuator testing (matches the sim test's net)."""

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(4, 1)

    def forward(
        self,
        x: torch.Tensor,
        hc: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        out, hc_new = self.lstm(x, hc)
        return self.fc(out[:, -1, :]), hc_new


def _make_dummy_lstm_checkpoint() -> str:
    """Create a deterministic TorchScript LSTM checkpoint with metadata."""
    torch.manual_seed(42)
    net = _DummyLSTM().eval()
    scripted = torch.jit.script(net)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    extra = {"metadata.json": json.dumps({"model_type": "lstm"})}
    torch.jit.save(scripted, tmp_path, _extra_files=extra)
    return tmp_path


class TestNeuralLSTMEquivalence(unittest.TestCase):
    """Lab vs Newton LSTM actuator on identical weights, driven directly."""

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device("cpu")
        cls.path = _make_dummy_lstm_checkpoint()

        # Lab actuator (pure torch; no articulation / sim app required).
        cfg = ActuatorNetLSTMCfg(
            joint_names_expr=[".*"],
            network_file=cls.path,
            saturation_effort=120.0,
            effort_limit=80.0,
            velocity_limit=7.5,
        )
        cls.lab = ActuatorNetLSTM(
            cfg,
            joint_names=[f"j{i}" for i in range(_NUM_JOINTS)],
            joint_ids=slice(None),
            num_envs=1,
            device="cpu",
        )

        # Newton controller from the same checkpoint.
        cls.newton = ControllerNeuralLSTM(cls.path)
        cls.newton.finalize(cls.device, _NUM_JOINTS)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.path)

    def _lab_effort(self, joint_pos: list[float], target_pos: list[float], joint_vel: list[float]) -> np.ndarray:
        """Raw (pre-clamp) network effort from the Lab actuator."""
        self.lab.reset([0])  # start from a zero hidden state
        action = ArticulationActions(joint_positions=torch.tensor([target_pos], dtype=torch.float32))
        self.lab.compute(
            action,
            torch.tensor([joint_pos], dtype=torch.float32),
            torch.tensor([joint_vel], dtype=torch.float32),
        )
        return self.lab.computed_effort.flatten().detach().cpu().numpy()

    def _newton_effort(
        self, joint_pos: list[float], target_pos: list[float], joint_vel: list[float], target_vel: list[float]
    ) -> np.ndarray:
        """Raw (pre-clamp) network effort from the Newton controller."""
        state = self.newton.state(_NUM_JOINTS, self.device)  # fresh zero hidden state
        idx = wp.array(np.arange(_NUM_JOINTS, dtype=np.uint32), dtype=wp.uint32, device=self.device)
        forces = wp.zeros(_NUM_JOINTS, dtype=wp.float32, device=self.device)
        self.newton.compute(
            wp.array(np.array(joint_pos, dtype=np.float32), dtype=wp.float32, device=self.device),
            wp.array(np.array(joint_vel, dtype=np.float32), dtype=wp.float32, device=self.device),
            wp.array(np.array(target_pos, dtype=np.float32), dtype=wp.float32, device=self.device),
            wp.array(np.array(target_vel, dtype=np.float32), dtype=wp.float32, device=self.device),
            None,  # feedforward
            idx,  # pos_indices
            idx,  # vel_indices
            idx,  # target_pos_indices (same object -> identity gather)
            idx,  # target_vel_indices
            forces,
            state,
            1.0 / 60.0,  # dt
            self.device,
        )
        return forces.numpy()

    def _assert_efforts_match(self, lab_effort: np.ndarray, newton_effort: np.ndarray, msg: str) -> None:
        torch.testing.assert_close(
            torch.from_numpy(lab_effort),
            torch.from_numpy(newton_effort),
            atol=1e-5,
            rtol=1e-5,
            msg=msg,
        )

    def test_output_matches_across_positions(self):
        """Zero velocity: both implementations produce the same effort at every position."""
        for joint_pos, target_pos in _ZERO_VELOCITY_CASES:
            self._assert_efforts_match(
                self._lab_effort(joint_pos, target_pos, _ZERO_VEL),
                self._newton_effort(joint_pos, target_pos, _ZERO_VEL, _ZERO_VEL),
                msg=f"effort mismatch at joint_pos={joint_pos}, target_pos={target_pos} (zero velocity)",
            )

    def test_output_matches_with_velocity(self):
        """Non-zero velocity (zero velocity target): efforts must still match."""
        for joint_pos, target_pos, joint_vel in _VELOCITY_CASES:
            self._assert_efforts_match(
                self._lab_effort(joint_pos, target_pos, joint_vel),
                self._newton_effort(joint_pos, target_pos, joint_vel, _ZERO_VEL),
                msg=f"effort mismatch at joint_pos={joint_pos}, target_pos={target_pos}, joint_vel={joint_vel}",
            )


if __name__ == "__main__":
    unittest.main()
