# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the versioned LEAPP/MotionGen interchange contract."""

import importlib
import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

leapp = pytest.importorskip("leapp")

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

    semantics = mg_semantics.estimated.signal(name="signal", tensor=tensor)
    description = TensorDescription("signal", tensor, semantics=semantics).dict()

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


@pytest.mark.parametrize(
    ("tensor", "names", "match"),
    [
        (torch.zeros((2, 2), dtype=torch.float32), ["joint_a", "joint_b"], "shape"),
        (torch.zeros((1, 3), dtype=torch.float32), ["joint_a", "joint_b"], "shape"),
        (torch.zeros((1, 2), dtype=torch.float64), ["joint_a", "joint_b"], "dtype"),
        (torch.zeros((1, 2), dtype=torch.float32), ["joint_a", "joint_a"], "unique"),
        (torch.zeros((1, 0), dtype=torch.float32), [], "non-empty"),
    ],
)
def test_joint_helpers_reject_invalid_contract(tensor: torch.Tensor, names: list[str], match: str) -> None:
    """Joint helpers reject invalid batch, joint-axis, dtype, and names."""
    with pytest.raises(ValueError, match=match):
        mg_semantics.estimated.joint_positions(tensor=tensor, names=names)


@pytest.mark.parametrize(
    ("factory", "tensor", "names", "kwargs"),
    [
        (mg_semantics.estimated.site_poses, torch.zeros((1, 2, 6)), ["base", "tool"], {}),
        (mg_semantics.estimated.site_poses, torch.zeros((2, 2, 7)), ["base", "tool"], {}),
        (mg_semantics.estimated.site_poses, torch.zeros((1, 1, 7)), ["base", "tool"], {}),
        (mg_semantics.estimated.site_poses, torch.zeros((1, 2, 7), dtype=torch.float64), ["base", "tool"], {}),
        (
            mg_semantics.estimated.site_twists,
            torch.zeros((1, 2, 7)),
            ["base", "tool"],
            {"expressed_in": "world"},
        ),
    ],
)
def test_site_helpers_reject_invalid_tensor_contract(
    factory: Callable[..., TensorSemantics], tensor: torch.Tensor, names: list[str], kwargs: dict
) -> None:
    """Site helpers reject invalid batch, site-axis, component-axis, and dtype layouts."""
    with pytest.raises(ValueError):
        factory(tensor=tensor, names=names, **kwargs)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros((), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.complex64),
    ],
)
def test_signal_helpers_reject_invalid_tensor_contract(tensor: torch.Tensor) -> None:
    """Signal helpers require one batch entry and a supported scalar dtype."""
    with pytest.raises(ValueError):
        mg_semantics.estimated.signal(name="signal", tensor=tensor)


def test_signal_helpers_reject_empty_name() -> None:
    """Signal helpers require a canonical non-empty signal name."""
    with pytest.raises(ValueError, match="non-empty"):
        mg_semantics.estimated.signal(name="", tensor=torch.zeros((1,), dtype=torch.bool))


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


def _load_source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_wip_motiongen_deploy() -> tuple[types.ModuleType, types.ModuleType]:
    try:
        motion_generation = importlib.import_module("isaacsim.robot_motion.experimental.motion_generation")
        leapp_deploy = importlib.import_module("isaacsim.robot_motion.leapp_deploy")
        return motion_generation, leapp_deploy
    except ModuleNotFoundError:
        pass

    isaac_sim_root = Path(__file__).resolve().parents[5] / "omni_isaac_sim"
    if not isaac_sim_root.is_dir():
        pytest.skip(f"WIP Isaac Sim checkout not found at {isaac_sim_root}")

    package_names = [
        "isaacsim",
        "isaacsim.robot_motion",
        "isaacsim.robot_motion.experimental",
        "isaacsim.robot_motion.experimental.motion_generation",
        "isaacsim.robot_motion.experimental.motion_generation.impl",
        "isaacsim.robot_motion.leapp_deploy",
        "isaacsim.robot_motion.leapp_deploy.impl",
    ]
    for name in package_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    motiongen_root = isaac_sim_root / "source/libraries/isaacsim/robot_motion/experimental/motion_generation"
    motiongen_impl = motiongen_root / "python/impl"
    _load_source_module(
        "isaacsim.robot_motion.experimental.motion_generation.impl._logging",
        motiongen_impl / "_logging.py",
    )
    types_module = _load_source_module(
        "isaacsim.robot_motion.experimental.motion_generation.impl.types",
        motiongen_impl / "types.py",
    )
    base_controller = _load_source_module(
        "isaacsim.robot_motion.experimental.motion_generation.impl.base_controller",
        motiongen_impl / "base_controller.py",
    )
    robot_spec_check = _load_source_module(
        "isaacsim.robot_motion.experimental.motion_generation.impl.robot_spec_check",
        motiongen_impl / "robot_spec_check.py",
    )
    motion_generation = sys.modules["isaacsim.robot_motion.experimental.motion_generation"]
    for symbol in ("JointState", "RobotSpec", "RobotState", "SignalKey", "SpatialState", "TwistFrame"):
        setattr(motion_generation, symbol, getattr(types_module, symbol))
    motion_generation.BaseController = base_controller.BaseController
    motion_generation.check_robot_spec = robot_spec_check.check_robot_spec

    deploy_impl = isaac_sim_root / "source/libraries/isaacsim/robot_motion/leapp_deploy/python/impl"
    _load_source_module("isaacsim.robot_motion.leapp_deploy.impl._logging", deploy_impl / "_logging.py")
    _load_source_module("isaacsim.robot_motion.leapp_deploy.impl.leapp_binding", deploy_impl / "leapp_binding.py")
    controller_module = _load_source_module(
        "isaacsim.robot_motion.leapp_deploy.impl.leapp_controller",
        deploy_impl / "leapp_controller.py",
    )
    leapp_deploy = sys.modules["isaacsim.robot_motion.leapp_deploy"]
    leapp_deploy.LeappController = controller_module.LeappController
    return motion_generation, leapp_deploy


def _export_motiongen_pipeline(root: Path) -> Path:
    policy_name = "motiongen_contract"
    joint_positions = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    site_poses = torch.tensor([[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    force = torch.tensor([2.0], dtype=torch.float32)

    leapp.start(policy_name, save_path=str(root), global_patching=False)
    try:
        joint_positions = leapp.annotate.input_tensors(
            policy_name,
            mg_semantics.estimated.joint_positions(tensor=joint_positions, names=["j0", "j1"]),
        )
        site_poses = leapp.annotate.input_tensors(
            policy_name,
            mg_semantics.estimated.site_poses(tensor=site_poses, names=["tool"]),
        )
        force = leapp.annotate.input_tensors(
            policy_name,
            mg_semantics.estimated.signal(name="force", tensor=force),
        )
        joint_targets = joint_positions + force.unsqueeze(1)
        site_targets = site_poses + 0.0
        vacuum = force > 0.0
        leapp.annotate.output_tensors(
            policy_name,
            [
                mg_semantics.desired.joint_positions(tensor=joint_targets, names=["j0", "j1"]),
                mg_semantics.desired.site_poses(tensor=site_targets, names=["tool"]),
                mg_semantics.desired.signal(name="vacuum", tensor=vacuum),
            ],
            export_with="jit-trace",
        )
    finally:
        leapp.stop()
    leapp.compile_graph(
        visualize=False,
        validate=False,
        graph_configs=GraphConfigs(extra={"motiongen_interface_version": 1}),
    )
    return root / policy_name / f"{policy_name}.yaml"


@pytest.mark.isaacsim_ci
def test_exported_helpers_match_motiongen_controller(tmp_path: Path) -> None:
    """Match raw LEAPP outputs after exporting through the Isaac Lab helpers."""
    motion_generation, leapp_deploy = _load_wip_motiongen_deploy()
    wp = importlib.import_module("warp")

    policy_path = _export_motiongen_pipeline(tmp_path)
    raw_runtime = leapp.InferenceManager(str(policy_path))
    controller = leapp_deploy.LeappController.from_export(
        policy_path,
        robot_spec=motion_generation.RobotSpec(
            joint_space=["j0", "j1"],
            site_space=["tool"],
            signals=[
                motion_generation.SignalKey("force", wp.float32),
                motion_generation.SignalKey("vacuum", wp.bool),
            ],
        ),
    )
    robot_spec = controller.robot_spec
    estimated = motion_generation.RobotState(
        robot_spec=robot_spec,
        joints=motion_generation.JointState.from_name(
            robot_joint_space=robot_spec.joint_space,
            positions=(["j0", "j1"], wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")),
        ),
        sites=motion_generation.SpatialState.from_name(
            spatial_space=robot_spec.site_space,
            positions=(["tool"], wp.array([[1.0, 2.0, 3.0]], dtype=wp.float32, device="cpu")),
            orientations=(["tool"], wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.float32, device="cpu")),
        ),
        signals={robot_spec.signals["force"]: wp.array([2.0], dtype=wp.float32, device="cpu")},
    )

    raw_runtime.set_input_value(
        node_name="motiongen_contract", input_name="estimated_joint_positions", value=torch.tensor([[1.0, 2.0]])
    )
    raw_runtime.set_input_value(
        node_name="motiongen_contract",
        input_name="estimated_site_poses",
        value=torch.tensor([[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]]),
    )
    raw_runtime.set_input_value(node_name="motiongen_contract", input_name="force", value=torch.tensor([2.0]))
    raw_outputs = raw_runtime({})
    desired = controller.forward(estimated, None, t=0.0)

    assert desired is not None
    controller_joints = torch.from_numpy(desired.joints.positions.numpy()).unsqueeze(0)
    controller_pose = torch.cat(
        (
            torch.from_numpy(desired.sites.positions.numpy()),
            torch.from_numpy(desired.sites.orientations.numpy()),
        ),
        dim=-1,
    ).unsqueeze(0)
    controller_vacuum = torch.from_numpy(desired.get(robot_spec.signals["vacuum"]).numpy())
    torch.testing.assert_close(
        controller_joints, raw_outputs["motiongen_contract/desired_joint_positions"].cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        controller_pose, raw_outputs["motiongen_contract/desired_site_poses"].cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(controller_vacuum, raw_outputs["motiongen_contract/vacuum"].cpu(), rtol=0, atol=0)
    assert controller_joints.tolist() == [[3.0, 4.0]]
    assert controller_pose.tolist() == [[[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]]
    assert controller_vacuum.tolist() == [True]
