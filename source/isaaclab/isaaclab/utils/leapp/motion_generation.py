# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MotionGen semantic helpers for LEAPP boundary tensors."""

from __future__ import annotations

from typing import Literal

import torch
from leapp import InputKindEnum, OutputKindEnum
from leapp.utils.tensor_description import TensorSemantics

__all__ = ["desired", "estimated", "setpoint", "site_pose_from_xyzw"]

_POSE_COMPONENTS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
_TWIST_COMPONENTS = ["vx", "vy", "vz", "wx", "wy", "wz"]
_SIGNAL_DTYPES = {
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
}


def _validate_names(*, names: list[str], label: str) -> None:
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{label} names must be non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique.")


def _validate_float_tensor(*, tensor: torch.Tensor, shape: tuple[int, ...], label: str) -> None:
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{label} tensor must have dtype float32, got {tensor.dtype}.")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{label} tensor must have shape {shape}, got {tuple(tensor.shape)}.")


def site_pose_from_xyzw(*, positions: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    """Pack positions and ``xyzw`` orientations into MotionGen ``xyz+wxyz`` site poses.

    Args:
        positions: Site positions [m], shape ``(..., 3)``.
        orientations: Site orientations as ``(qx, qy, qz, qw)``, shape ``(..., 4)``.

    Returns:
        Site poses ordered as ``(x, y, z, qw, qx, qy, qz)``, shape ``(..., 7)``.
    """
    if positions.shape[-1:] != (3,):
        raise ValueError(f"positions must have shape (..., 3), got {tuple(positions.shape)}.")
    if orientations.shape[-1:] != (4,):
        raise ValueError(f"orientations must have shape (..., 4), got {tuple(orientations.shape)}.")
    if positions.shape[:-1] != orientations.shape[:-1]:
        raise ValueError("positions and orientations must have matching leading dimensions.")

    orientations_wxyz = torch.cat((orientations[..., 3:4], orientations[..., :3]), dim=-1)
    return torch.cat((positions, orientations_wxyz), dim=-1)


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
        if not isinstance(name, str) or not name:
            raise ValueError("Signal name must be a non-empty string.")
        if tensor.ndim < 1 or tensor.shape[0] != 1:
            raise ValueError(f"Signal tensor must have leading batch size 1, got shape {tuple(tensor.shape)}.")
        if tensor.dtype not in _SIGNAL_DTYPES:
            raise ValueError(f"Signal tensor has unsupported scalar dtype {tensor.dtype}.")
        return TensorSemantics(
            name=name,
            ref=tensor,
            kind=f"motiongen/{self._role}/signal",
            extra={"motiongen_signal": name},
        )

    def site_poses(self, *, tensor: torch.Tensor, names: list[str]) -> TensorSemantics:
        """Describe World-to-Site poses in MotionGen ``xyz+wxyz`` order."""
        _validate_names(names=names, label="Site")
        _validate_float_tensor(tensor=tensor, shape=(1, len(names), 7), label="Site pose")
        return TensorSemantics(
            name=f"{self._role}_site_poses",
            ref=tensor,
            kind=f"motiongen/{self._role}/site_pose",
            element_names=[names, _POSE_COMPONENTS],
        )

    def site_twists(
        self,
        *,
        tensor: torch.Tensor,
        names: list[str],
        expressed_in: Literal["world", "site"],
    ) -> TensorSemantics:
        """Describe site twists ordered as linear then angular velocity."""
        _validate_names(names=names, label="Site")
        _validate_float_tensor(tensor=tensor, shape=(1, len(names), 6), label="Site twist")
        if expressed_in not in ("world", "site"):
            raise ValueError(f"expressed_in must be 'world' or 'site', got {expressed_in!r}.")
        return TensorSemantics(
            name=f"{self._role}_site_twists",
            ref=tensor,
            kind=f"motiongen/{self._role}/site_twist",
            element_names=[names, _TWIST_COMPONENTS],
            extra={"motiongen_twist_frame": expressed_in},
        )

    def _joint(
        self,
        *,
        tensor: torch.Tensor,
        names: list[str],
        field: str,
        kind: InputKindEnum | OutputKindEnum,
    ) -> TensorSemantics:
        _validate_names(names=names, label="Joint")
        _validate_float_tensor(tensor=tensor, shape=(1, len(names)), label="Joint")
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
