# Newton Neural Actuators: Trace & Velocity-Channel Mismatch

Isaac Lab commit:  
e5478465baf58f735ba308e40faf6b4eb2993d29

## Summary

Isaac Lab users still construct the usual Lab actuator configs (`ActuatorNetMLPCfg`,
`ActuatorNetLSTMCfg`, etc.). When `SimulationCfg.use_newton_actuators=True`, those configs
are serialized to `NewtonActuator` USD prims at spawn time, then reconstructed as
`newton.actuators.Actuator`.

The metadata bridge reconciles input ordering, history indices, and scale factors. But the
**velocity input channel differs semantically**: Lab feeds *raw joint velocity*, while the
Newton neural controllers feed *velocity error* (`target_vel - vel`). With the default zero
velocity target, Newton feeds `-joint_vel` — sign-inverted relative to what the network
expects.

Paths below are relative to the repo root. `newton` paths refer to the 1.2.1 RC prebundle
(see [Caveat](#caveat)).

## A. Lab-native reference (what the network was trained to see)

`source/isaaclab/isaaclab/actuators/actuator_net.py`

- MLP `152` — `self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos` (position **error**)
- MLP `155` — `self._joint_vel_history[:, 0] = joint_vel` — **raw velocity**
- MLP `167-170` — `input_order` (pos_vel/vel_pos) with `pos_scale` / `vel_scale`
- MLP `179` — output scaled by `cfg.torque_scale`
- LSTM `81` — `sea_input[:, 0, 0] = control_action.joint_positions - joint_pos` (position **error**)
- LSTM `82` — `sea_input[:, 0, 1] = joint_vel` — **raw velocity**

## B. Config -> USD authoring (Lab side)

- `source/isaaclab/isaaclab/sim/simulation_cfg.py:87` — `use_newton_actuators: bool = False` (opt-in gate)
- `source/isaaclab/isaaclab/assets/articulation/articulation_cfg.py:78,98` — `_post_spawn()` -> `define_actuator_properties(...)`
- `source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py`
  - `119` — gate: returns unless `use_newton_actuators`
  - `155` — implicit actuators skipped
  - `196` — `is_neural = isinstance(cfg, (ActuatorNetMLPCfg, ActuatorNetLSTMCfg))`
  - `211-216` — writes metadata: `model_type`, `input_order`, `input_idx`, `pos_scale`, `vel_scale`, `torque_scale`
  - `219` — `_resave_checkpoint_with_metadata(...)` bakes metadata into the checkpoint
  - `229` — attaches `NewtonNeuralControlAPI` schema to the prim

## C. USD -> `newton.actuators.Actuator` construction

- `source/isaaclab_newton/isaaclab_newton/actuators/adapter.py`
  - `26` — `from newton.actuators import Actuator, Clamping, Delay`
  - `190` — `from_usd(...)` (PhysX path) -> `335` `_create_actuators_from_usd(...)`
  - `366,381` — `parse_actuator_prim(prim)`
  - `448-450` — `Actuator(controller=controller, ...)`
- `source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:1947,1972` — `activate_newton_actuator_path()` builds the single `NewtonActuatorAdapter(...)`

## D. Runtime orchestration (Lab-Newton articulation)

`source/isaaclab_newton/isaaclab_newton/assets/articulation/articulation.py`

- `3512` — `_process_actuators_cfg()` called during init
- `3646` — reads `use_newton_actuators`
- `3665,3668` — sets `_has_newton_actuators = True`, calls `activate_newton_actuator_path()`
- `3704` — still creates Lab actuator objects with `properties_only=True`
- `3555` — `self._joint_vel_target_sim = wp.zeros_like(...)` — **velocity target defaults to 0**
- `3942` — feeds `joint_velocities=self._data.joint_vel_target...` into the actuator
- Adapter step: `adapter.py:110,133` — `step()` forwards `sim_control` into `act.step(...)`

## E. Consumer side (external `newton`)

Prebundle root:
`~/.cache/packman/chk/repo_pip_cache/59ba294c6533f3fb066ebad2c663408baad2629a-v2-linux-x86_64-licensed/isaac_newton_prebundle/newton`

- `_src/actuators/utils.py:21-24,52` — `load_checkpoint()` reads `metadata.json` from the TorchScript `_extra_files` (consumes the B/`219` write)
- `_src/actuators/controllers/controller_neural_mlp.py`
  - `92,96,101-103` — reads `input_order`, `input_idx`, `pos_scale`, `vel_scale`, `effort_scale` (falls back to `torque_scale`)
  - `166` — `pos_error = target_p[...] - current_pos[...]` — matches Lab
  - `167` — `vel_error = target_v[...] - current_vel[...]` — **velocity error, not raw velocity**
  - `179-185` — assembles input (order/scales) and runs the net
- `_src/actuators/controllers/controller_neural_lstm.py`
  - `108` — asserts `lstm.input_size == 2` with features `[pos_error, vel_error]`
  - `178` — `pos_error = target_p - current_pos` — matches Lab
  - `179` — `vel_error = target_v - current_vel` — **velocity error**
  - `182` — `net_input = stack([pos_error*pos_scale, vel_error*vel_scale])`

## The pinch point

| Channel | Lab (A) | Newton (E) | Match? |
|---|---|---|---|
| Position | `target_pos - pos` (`actuator_net.py:152,81`) | `target_p - current_pos` (`:166,178`) | yes |
| Velocity | raw `joint_vel` (`actuator_net.py:155,82`) | `target_vel - vel` (`:167,179`) | no |

With the default zero velocity target (`articulation.py:3555`), Newton feeds
`0 - joint_vel = -joint_vel` into the second channel, whereas Lab feeds `+joint_vel`. The
metadata bridge does not compensate (same `vel_scale` scalar on both sides, no sign flip).

## Caveat

Section E reflects the newton 1.2.1 RC prebundle. The `env_isaaclab` newton is 1.0.0 and has
no `actuators` module, so confirm which newton build the running Isaac Sim actually loads
before treating these exact line numbers as authoritative for your runtime.
