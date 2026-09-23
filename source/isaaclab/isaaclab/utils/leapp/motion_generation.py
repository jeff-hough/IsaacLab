# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MotionGen semantic helpers for LEAPP boundary tensors."""

from __future__ import annotations

import torch
from leapp import InputKindEnum, OutputKindEnum
from leapp.utils.tensor_description import TensorSemantics

__all__ = ["desired", "estimated", "setpoint"]


class _StateSemantics:
    """Construct LEAPP tensor semantics for one MotionGen state role."""

    def __init__(
        self,
        *,
        role: str,
        position_kind: InputKindEnum | OutputKindEnum,
        velocity_kind: InputKindEnum | OutputKindEnum,
        effort_kind: InputKindEnum | OutputKindEnum,
    ) -> None:
        self._role = role
        self._position_kind = position_kind
        self._velocity_kind = velocity_kind
        self._effort_kind = effort_kind

    def joint_positions(self, *, tensor: torch.Tensor, names: list[str]) -> TensorSemantics:
        """Describe a joint-position tensor."""
        return self._joint(tensor=tensor, names=names, field="positions", kind=self._position_kind)

    def joint_velocities(self, *, tensor: torch.Tensor, names: list[str]) -> TensorSemantics:
        """Describe a joint-velocity tensor."""
        return self._joint(tensor=tensor, names=names, field="velocities", kind=self._velocity_kind)

    def joint_efforts(self, *, tensor: torch.Tensor, names: list[str]) -> TensorSemantics:
        """Describe a joint-effort tensor."""
        return self._joint(tensor=tensor, names=names, field="efforts", kind=self._effort_kind)

    def signal(self, *, name: str, tensor: torch.Tensor) -> TensorSemantics:
        """Describe a tensor backed by a MotionGen ``SignalKey``."""
        return TensorSemantics(
            name=name,
            ref=tensor,
            kind=f"motiongen/{self._role}/signal",
            extra={"motiongen_signal": name},
        )

    def _joint(
        self,
        *,
        tensor: torch.Tensor,
        names: list[str],
        field: str,
        kind: InputKindEnum | OutputKindEnum,
    ) -> TensorSemantics:
        return TensorSemantics(
            name=f"{self._role}_joint_{field}",
            ref=tensor,
            kind=kind,
            element_names=names,
        )


estimated = _StateSemantics(
    role="estimated",
    position_kind=InputKindEnum.JOINT_POSITION,
    velocity_kind=InputKindEnum.JOINT_VELOCITY,
    effort_kind=InputKindEnum.JOINT_EFFORT,
)

setpoint = _StateSemantics(
    role="setpoint",
    position_kind=InputKindEnum.COMMAND_JOINT_POSITION,
    velocity_kind=InputKindEnum.COMMAND_JOINT_VELOCITY,
    effort_kind=InputKindEnum.COMMAND_JOINT_TORQUES,
)

desired = _StateSemantics(
    role="desired",
    position_kind=OutputKindEnum.JOINT_POSITION,
    velocity_kind=OutputKindEnum.JOINT_VELOCITY,
    effort_kind=OutputKindEnum.JOINT_EFFORT,
)
