# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the versioned LEAPP/MotionGen interchange contract."""

from collections.abc import Callable

import pytest
import torch

pytest.importorskip("leapp")

from leapp import GraphConfigs, InputKindEnum, OutputKindEnum
from leapp.utils.tensor_description import TensorDescription, TensorSemantics

from isaaclab.utils.leapp import motion_generation as mg_semantics


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


@pytest.mark.parametrize(
    ("factory", "kind", "name"),
    [
        (mg_semantics.estimated.joint_positions, InputKindEnum.JOINT_POSITION, "estimated_joint_positions"),
        (mg_semantics.estimated.joint_velocities, InputKindEnum.JOINT_VELOCITY, "estimated_joint_velocities"),
        (mg_semantics.estimated.joint_efforts, InputKindEnum.JOINT_EFFORT, "estimated_joint_efforts"),
        (mg_semantics.setpoint.joint_positions, InputKindEnum.COMMAND_JOINT_POSITION, "setpoint_joint_positions"),
        (mg_semantics.setpoint.joint_velocities, InputKindEnum.COMMAND_JOINT_VELOCITY, "setpoint_joint_velocities"),
        (mg_semantics.setpoint.joint_efforts, InputKindEnum.COMMAND_JOINT_TORQUES, "setpoint_joint_efforts"),
        (mg_semantics.desired.joint_positions, OutputKindEnum.JOINT_POSITION, "desired_joint_positions"),
        (mg_semantics.desired.joint_velocities, OutputKindEnum.JOINT_VELOCITY, "desired_joint_velocities"),
        (mg_semantics.desired.joint_efforts, OutputKindEnum.JOINT_EFFORT, "desired_joint_efforts"),
    ],
)
def test_joint_helpers(
    factory: Callable[..., TensorSemantics], kind: InputKindEnum | OutputKindEnum, name: str
) -> None:
    """Joint helpers select the role-specific kind, name, and ordering."""
    tensor = torch.zeros((1, 2), dtype=torch.float32)

    semantics = factory(tensor=tensor, names=["joint_b", "joint_a"])

    assert semantics.name == name
    assert semantics.ref is tensor
    assert semantics.kind is kind
    assert semantics.element_names == [["joint_b", "joint_a"]]


@pytest.mark.parametrize(
    ("factory", "kind"),
    [
        (mg_semantics.estimated.signal, "motiongen/estimated/signal"),
        (mg_semantics.setpoint.signal, "motiongen/setpoint/signal"),
        (mg_semantics.desired.signal, "motiongen/desired/signal"),
    ],
)
def test_signal_helpers(factory: Callable[..., TensorSemantics], kind: str) -> None:
    """Signal helpers attach the role and canonical SignalKey name."""
    tensor = torch.zeros((1,), dtype=torch.bool)

    semantics = factory(name="flange.vacuum", tensor=tensor)

    assert semantics.name == "flange.vacuum"
    assert semantics.ref is tensor
    assert semantics.kind == kind
    assert semantics.extra == {"motiongen_signal": "flange.vacuum"}


def test_site_pose_from_xyzw() -> None:
    """Lab-native xyzw poses are explicitly packed in MotionGen order."""
    positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    orientations = torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])

    poses = mg_semantics.site_pose_from_xyzw(positions=positions, orientations=orientations)

    torch.testing.assert_close(
        poses,
        torch.tensor([[[1.0, 2.0, 3.0, 0.4, 0.1, 0.2, 0.3], [4.0, 5.0, 6.0, 0.8, 0.5, 0.6, 0.7]]]),
    )


@pytest.mark.parametrize(
    ("factory", "kind"),
    [
        (mg_semantics.estimated.site_poses, "motiongen/estimated/site_pose"),
        (mg_semantics.setpoint.site_poses, "motiongen/setpoint/site_pose"),
        (mg_semantics.desired.site_poses, "motiongen/desired/site_pose"),
    ],
)
def test_site_pose_helpers(factory: Callable[..., TensorSemantics], kind: str) -> None:
    """Site-pose helpers record site names and canonical component order."""
    tensor = torch.zeros((1, 2, 7), dtype=torch.float32)

    description = TensorDescription(
        "site_poses", tensor, semantics=factory(tensor=tensor, names=["base", "tool"])
    ).dict()

    assert description["kind"] == kind
    assert description["element_names"] == [
        ["base", "tool"],
        ["x", "y", "z", "qw", "qx", "qy", "qz"],
    ]


@pytest.mark.parametrize(
    ("factory", "kind"),
    [
        (mg_semantics.estimated.site_twists, "motiongen/estimated/site_twist"),
        (mg_semantics.setpoint.site_twists, "motiongen/setpoint/site_twist"),
        (mg_semantics.desired.site_twists, "motiongen/desired/site_twist"),
    ],
)
def test_site_twist_helpers(factory: Callable[..., TensorSemantics], kind: str) -> None:
    """Site-twist helpers record component order and expression frame."""
    tensor = torch.zeros((1, 1, 6), dtype=torch.float32)

    description = TensorDescription(
        "site_twists",
        tensor,
        semantics=factory(tensor=tensor, names=["base"], expressed_in="site"),
    ).dict()

    assert description["kind"] == kind
    assert description["element_names"] == [["base"], ["vx", "vy", "vz", "wx", "wy", "wz"]]
    assert description["motiongen_twist_frame"] == "site"


def test_site_helpers_reject_invalid_arguments() -> None:
    """Pose shapes and twist-frame values are validated explicitly."""
    with pytest.raises(ValueError, match="matching leading dimensions"):
        mg_semantics.site_pose_from_xyzw(positions=torch.zeros((1, 2, 3)), orientations=torch.zeros((1, 1, 4)))
    with pytest.raises(ValueError, match="expressed_in"):
        mg_semantics.estimated.site_twists(tensor=torch.zeros((1, 1, 6)), names=["base"], expressed_in="body")


def test_helpers_require_keyword_arguments() -> None:
    """Public helper arguments are keyword-only."""
    tensor = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(TypeError):
        mg_semantics.estimated.joint_positions(tensor, ["joint_a", "joint_b"])
    with pytest.raises(TypeError):
        mg_semantics.estimated.signal("terrain.height_scan", tensor)
    with pytest.raises(TypeError):
        mg_semantics.site_pose_from_xyzw(torch.zeros((1, 3)), torch.zeros((1, 4)))
