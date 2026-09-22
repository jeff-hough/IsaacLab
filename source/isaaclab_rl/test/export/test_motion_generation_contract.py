# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the versioned LEAPP/MotionGen interchange contract."""

import pytest
import torch

pytest.importorskip("leapp")

from leapp import GraphConfigs, InputKindEnum, OutputKindEnum
from leapp.utils.tensor_description import TensorDescription, TensorSemantics


def test_standard_joint_semantic_kinds() -> None:
    """Standard LEAPP joint kinds retain the values consumed by MotionGen."""
    assert InputKindEnum.JOINT_POSITION.value == "state/joint/position"
    assert InputKindEnum.JOINT_VELOCITY.value == "state/joint/velocity"
    assert InputKindEnum.JOINT_EFFORT.value == "state/joint/effort"
    assert InputKindEnum.COMMAND_JOINT_POSITION.value == "command/joint/position"
    assert InputKindEnum.COMMAND_JOINT_VELOCITY.value == "command/joint/velocity"
    assert InputKindEnum.COMMAND_JOINT_TORQUES.value == "command/joint/torques"
    assert OutputKindEnum.JOINT_POSITION.value == "target/joint/position"
    assert OutputKindEnum.JOINT_VELOCITY.value == "target/joint/velocity"
    assert OutputKindEnum.JOINT_EFFORT.value == "target/joint/effort"


@pytest.mark.parametrize(
    ("kind", "signal_name", "tensor"),
    [
        ("motiongen/estimated/signal", "terrain.height_scan", torch.zeros((1, 187), dtype=torch.float32)),
        ("motiongen/setpoint/signal", "command.base_velocity", torch.zeros((1, 3), dtype=torch.float32)),
        ("motiongen/desired/signal", "flange.vacuum", torch.zeros((1,), dtype=torch.bool)),
    ],
)
def test_signal_semantics_serialize(kind: str, signal_name: str, tensor: torch.Tensor) -> None:
    """Signal role and identity are serialized into the tensor description."""
    semantics = TensorSemantics(
        name="signal",
        ref=tensor,
        kind=kind,
        extra={"motiongen_signal": signal_name},
    )

    description = TensorDescription("signal", tensor, semantics=semantics).dict()

    assert description["kind"] == kind
    assert description["motiongen_signal"] == signal_name
    assert description["dtype"] == str(tensor.dtype).removeprefix("torch.")
    assert description["shape"] == list(tensor.shape)


@pytest.mark.parametrize(
    ("torch_dtype", "leapp_dtype"),
    [
        (torch.bool, "bool"),
        (torch.int8, "int8"),
        (torch.int16, "int16"),
        (torch.int32, "int32"),
        (torch.int64, "int64"),
        (torch.float16, "float16"),
        (torch.float32, "float32"),
        (torch.float64, "float64"),
    ],
)
def test_signal_scalar_dtypes_serialize(torch_dtype: torch.dtype, leapp_dtype: str) -> None:
    """Supported scalar dtypes retain their canonical LEAPP spelling."""
    tensor = torch.zeros((1, 3), dtype=torch_dtype)

    description = TensorDescription("signal", tensor).dict()

    assert description["dtype"] == leapp_dtype


def test_joint_names_serialize_as_one_logical_axis() -> None:
    """Joint ordering is retained independently of the leading batch dimension."""
    tensor = torch.zeros((1, 2), dtype=torch.float32)
    semantics = TensorSemantics(
        name="joint_positions",
        ref=tensor,
        kind=InputKindEnum.JOINT_POSITION,
        element_names=["joint_b", "joint_a"],
    )

    description = TensorDescription("joint_positions", tensor, semantics=semantics).dict()

    assert description["element_names"] == [["joint_b", "joint_a"]]


def test_motiongen_interface_version_serializes() -> None:
    """The graph config carries the MotionGen interface version."""
    configs = GraphConfigs(extra={"motiongen_interface_version": 1})

    assert configs.to_dict() == {"motiongen_interface_version": 1}
