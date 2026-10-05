from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import osqp
from scipy import sparse
from loguru import logger
import pinocchio as pin

Array = np.ndarray

UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


##Angle wrapping
def wrap_to_pi(x: Array) -> Array:
    return np.arctan2(np.sin(x), np.cos(x))


@dataclass(frozen=True)
class PoseSample:
    position: Array
    rotation: Array
    twist: Array


@dataclass(frozen=True)
class DistanceSample:
    """One robot-feature to obstacle distance query in the world frame."""

    distance: float
    normal: Array #direction from the obstacle to the robot feature, normalized
    gradient: Array #derivative of the distance with respect to the robot joint angles
    feature_name: str #name of the robot feature (link or tool) that is closest to the obstacle
    point_robot: Array
    point_obstacle: Array


class StaticBoxCollision:
    """URDF link meshes plus a protective tool sphere against one static box.

    Coal/HPP-FCL computes the closest witness points.  The returned distance
    gradient is n.T @ J_witness, which is valid while the closest feature does
    not change."""

    def __init__(
        self,
        robot: "UR3ePinocchio",
        urdf_path: str,
        center: Array,
        size: Array,
        tool_radius: float,
    ) -> None:
        try:
            import coal
        except ImportError as exc:
            raise RuntimeError(
                "Static collision checking requires the coal Python package"
            ) from exc
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.size = np.asarray(size, dtype=float).reshape(3)
        if np.any(self.size <= 0.0) or tool_radius <= 0.0:
            raise ValueError("Obstacle side lengths and tool radius must be positive")

        self.robot = robot
        self.tool_radius = float(tool_radius)
        self.geometry_model = pin.buildGeomFromUrdf(
            robot.model, urdf_path, pin.GeometryType.COLLISION
        )

        # The fixed base cannot move away from an obstacle.  It is excluded from
        # MPC constraints; the six moving link collision meshes are retained.
        self.robot_geometry_ids = [
            gid
            for gid, geometry in enumerate(self.geometry_model.geometryObjects)
            if geometry.parentJoint != 0
        ]

        # tool0 has no URDF collision mesh.  Add a conservative sphere attached
        # to its parent joint so that the end effector is also protected.
        ee_frame = robot.model.frames[robot.frame_id]
        tool_geometry = pin.GeometryObject(
            "tool0_collision_sphere",
            ee_frame.parentJoint,
            robot.frame_id,
            ee_frame.placement,
            coal.Sphere(float(tool_radius)),
        )
        tool_gid = self.geometry_model.addGeometryObject(tool_geometry)
        self.robot_geometry_ids.append(tool_gid)
        # ``coal.Box`` takes full side lengths, not half-extents.  It is fixed
        # in the world frame, centred at ``self.center`` and axis aligned.
        obstacle_geometry = pin.GeometryObject(
            "static_box_obstacle",
            0,
            0,
            pin.SE3(np.eye(3), self.center),
            coal.Box(*self.size),
        )
        # Create a collision pair between each moving robot feature and box.
        self.obstacle_gid = self.geometry_model.addGeometryObject(obstacle_geometry)
        self.pair_ids: list[int] = []
        for robot_gid in self.robot_geometry_ids:
            self.geometry_model.addCollisionPair(
                pin.CollisionPair(robot_gid, self.obstacle_gid)
            )
            self.pair_ids.append(len(self.geometry_model.collisionPairs) - 1)
        self.geometry_data = self.geometry_model.createData()

    def query(self, theta: Array) -> list[DistanceSample]:
        """Return mesh/tool clearance, separating normal, and dd/dtheta."""
        q = self.robot.theta_to_q(theta)
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        pin.updateGeometryPlacements(
            self.robot.model,
            self.robot.data,
            self.geometry_model,
            self.geometry_data,
            q,
        )
        pin.computeDistances(
            self.robot.model,
            self.robot.data,
            self.geometry_model,
            self.geometry_data,
            q,
        )

        samples: list[DistanceSample] = []
        for pair_id in self.pair_ids:
            pair = self.geometry_model.collisionPairs[pair_id]
            result = self.geometry_data.distanceResults[pair_id]
            distance = float(result.min_distance)
            if not np.isfinite(distance):
                raise RuntimeError("GJK returned a non-finite link-obstacle distance")

            point_robot = np.asarray(result.getNearestPoint1(), dtype=float) #distance from the robot feature to the obstacle
            point_obstacle = np.asarray(result.getNearestPoint2(), dtype=float) #distance from the obstacle to the robot feature
            separation = point_robot - point_obstacle
            separation_norm = np.linalg.norm(separation) 
            if separation_norm > 1.0e-9:
                normal = separation / separation_norm ##direction from the obstacle to the robot feature, normalized
            else:
                # At contact the GJK normal is not unique.  This deterministic
                # fallback is only for reporting/linearisation; the positive
                # safety margin should keep normal operation away from contact.
                normal = np.array([1.0, 0.0, 0.0])

            robot_gid = pair.first
            geometry = self.geometry_model.geometryObjects[robot_gid]
            joint_id = geometry.parentJoint
            joint_origin = self.robot.data.oMi[joint_id].translation
            joint_jacobian = pin.getJointJacobian(
                self.robot.model,
                self.robot.data,
                joint_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            witness_jacobian = (
                joint_jacobian[:3]
                - pin.skew(point_robot - joint_origin) @ joint_jacobian[3:] ##Jpr
            )
            gradient = normal @ witness_jacobian[:, self.robot.velocity_indices] #n^T.Jpr
            samples.append(
                DistanceSample(
                    distance,
                    normal,
                    gradient,
                    geometry.name,
                    point_robot,
                    point_obstacle,
                )
            )
        return samples

    def minimum_distance(self, theta: Array) -> float:
        return min(sample.distance for sample in self.query(theta))

    def gradient_check(self, theta: Array, step: float = 1.0e-6) -> dict[str, float]:
        """Return per-feature max errors against central finite differences."""
        base = {sample.feature_name: sample for sample in self.query(theta)}
        numerical = {
            name: np.zeros(6) for name in base
        }
        for joint in range(6):
            perturbation = np.zeros(6)
            perturbation[joint] = step
            plus = {
                sample.feature_name: sample
                for sample in self.query(theta + perturbation)
            }
            minus = {
                sample.feature_name: sample
                for sample in self.query(theta - perturbation)
            }
            for name in base:
                numerical[name][joint] = (
                    plus[name].distance - minus[name].distance
                ) / (2.0 * step)
        return {
            name: float(np.max(np.abs(sample.gradient - numerical[name])))
            for name, sample in base.items()
        } ##this functions returns the maximum error between the analytical gradient 
    ##and the numerical gradient for each feature in the robot's geometry.
    # It uses central finite differences to compute the numerical gradient
    # by perturbing each joint angle slightly and measuring the change in distance to the obstacle. 


class MeshcatSimulationViewer:
    """Live simulation visualisation; it is never a safety-control component."""

    def __init__(
        self,
        robot: "UR3ePinocchio",
        collision: Optional[StaticBoxCollision],
        target_position: Array,
    ) -> None:
        import meshcat.geometry as geometry
        import meshcat.transformations as transformations
        from pinocchio.visualize import MeshcatVisualizer

        self.robot = robot
        self.collision = collision
        self.target_position = np.asarray(target_position, dtype=float).reshape(3)
        self.geometry = geometry
        self.transformations = transformations
        visual_model = pin.buildGeomFromUrdf(
            robot.model, robot.urdf_path, pin.GeometryType.VISUAL
        )
        self.visualizer = MeshcatVisualizer(
            robot.model, pin.GeometryModel(), visual_model
        )
        try:
            self.visualizer.initViewer(open=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Meshcat could not start its local viewer server. Run this from "
                "a normal local terminal with localhost socket access."
            ) from exc
        self.visualizer.loadViewerModel(rootNodeName="ur3e_mpc")
        self.viewer = self.visualizer.viewer

        self.tool_node = self.viewer["ur3e_mpc/tool0_collision_sphere"]
        tool_radius = collision.tool_radius if collision is not None else 0.03
        self.tool_node.set_object(
            geometry.Sphere(tool_radius),
            geometry.MeshPhongMaterial(
                color=0x1F77B4, opacity=0.35, transparent=True
            ),
        )
        self.target_node = self.viewer["ur3e_mpc/target_tool_position"]
        self.target_node.set_object(
            geometry.Sphere(0.012),
            geometry.MeshPhongMaterial(color=0xFFD700),
        )
        self.target_node.set_transform(
            transformations.translation_matrix(self.target_position)
        )
        if collision is not None:
            self.obstacle_node = self.viewer["ur3e_mpc/static_obstacle"]
            self.obstacle_node.set_object(
                geometry.Box(collision.size),
                geometry.MeshPhongMaterial(
                    color=0xD62728, opacity=0.65, transparent=True
                ),
            )
            self.obstacle_node.set_transform(
                transformations.translation_matrix(collision.center)
            )
            marker_material = geometry.MeshPhongMaterial(color=0x2CA02C)
            self.robot_witness_node = self.viewer["ur3e_mpc/witness_robot"]
            self.obstacle_witness_node = self.viewer["ur3e_mpc/witness_obstacle"]
            self.robot_witness_node.set_object(geometry.Sphere(0.008), marker_material)
            self.obstacle_witness_node.set_object(
                geometry.Sphere(0.008), marker_material
            )
        else:
            self.robot_witness_node = None
            self.obstacle_witness_node = None

        print("Meshcat viewer started. Close its browser tab when finished.")

    def display(self, theta: Array) -> None:
        self.visualizer.display(self.robot.theta_to_q(theta))
        tool_pose = self.robot.pose(theta)
        self.tool_node.set_transform(tool_pose.homogeneous)
        if self.collision is None:
            return

        nearest = min(self.collision.query(theta), key=lambda sample: sample.distance)
        self.robot_witness_node.set_transform(
            self.transformations.translation_matrix(nearest.point_robot)
        )
        self.obstacle_witness_node.set_transform(
            self.transformations.translation_matrix(nearest.point_obstacle)
        )

    def play_trajectory(self, theta_trajectory: Array, dt: float) -> None:
        """Install and automatically play a browser-side recording of the run."""
        from meshcat.animation import Animation

        animation = Animation(default_framerate=1.0 / dt)
        for frame, theta in enumerate(theta_trajectory):
            q = self.robot.theta_to_q(theta)
            pin.forwardKinematics(self.robot.model, self.visualizer.data, q)
            pin.updateGeometryPlacements(
                self.robot.model,
                self.visualizer.data,
                self.visualizer.visual_model,
                self.visualizer.visual_data,
            )
            with animation.at_frame(self.viewer, frame) as frame_viewer:
                for visual in self.visualizer.visual_model.geometryObjects:
                    geometry_id = self.visualizer.visual_model.getGeometryId(visual.name)
                    placement = self.visualizer.visual_data.oMg[geometry_id]
                    node_name = self.visualizer.getViewerNodeName(
                        visual, pin.GeometryType.VISUAL
                    )
                    frame_viewer[node_name].set_transform(placement.homogeneous)

                tool_pose = self.robot.pose(theta)
                frame_viewer["ur3e_mpc/tool0_collision_sphere"].set_transform(
                    tool_pose.homogeneous
                )
                if self.collision is not None:
                    nearest = min(
                        self.collision.query(theta),
                        key=lambda sample: sample.distance,
                    )
                    frame_viewer["ur3e_mpc/witness_robot"].set_transform(
                        self.transformations.translation_matrix(nearest.point_robot)
                    )
                    frame_viewer["ur3e_mpc/witness_obstacle"].set_transform(
                        self.transformations.translation_matrix(
                            nearest.point_obstacle
                        )
                    )
        # repetitions=1000 effectively loops for normal interactive use.
        self.viewer.set_animation(animation, play=True, repetitions=1000)

    def save_gif(self, theta_trajectory: Array, dt: float, output_path: str) -> Path:
        """Capture the recorded simulation as a looping GIF through Meshcat."""
        from PIL import Image

        path = Path(output_path).expanduser().resolve()
        if path.suffix.lower() != ".gif":
            raise ValueError("--save-meshcat-gif output must have a .gif suffix")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing GIF: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(f"GIF output directory does not exist: {path.parent}")

        frames: list[Image.Image] = []
        for theta in theta_trajectory:
            self.display(theta)
            image = Image.fromarray(self.visualizer.captureImage()).convert("RGB")
            frames.append(image)
        if not frames:
            raise RuntimeError("Cannot save an empty Meshcat trajectory")
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, round(1.0e3 * dt)),
            loop=0,
            disposal=2,
        )
        return path

###robot-model class using pinocchio
class UR3ePinocchio:
    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "tool0",
        joint_names: Sequence[str] = UR_JOINT_NAMES,
    ) -> None:
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF not found: {path}")

        self.urdf_path = str(path)
        self.model = pin.buildModelFromUrdf(str(object= path))
        self.data = self.model.createData()
        self.ee_frame = ee_frame
        self.frame_id = self.model.getFrameId(ee_frame)
        if self.frame_id >= self.model.nframes:
            available = ", ".join(frame.name for frame in self.model.frames)
            raise ValueError(
                f"Frame {ee_frame!r} is not in the URDF. Frames: {available}"
            )

        self.joint_names = tuple(joint_names)
        self.joint_ids = []
        for name in self.joint_names:
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(f"Required UR joint {name!r} is not in the URDF")
            if self.model.joints[jid].nv != 1:
                raise ValueError(f"Joint {name!r} must have one velocity DoF")
            self.joint_ids.append(jid)

        self.velocity_indices = np.array(
            [self.model.joints[jid].idx_v for jid in self.joint_ids], dtype=int
        )
        if len(set(self.velocity_indices.tolist())) != 6:
            raise ValueError("UR joint velocity indices are not unique")

        self.q_neutral = pin.neutral(self.model)
        self.lower, self.upper = self._joint_angle_limits()

        print(
            f"Loaded {path.name}: nq={self.model.nq}, nv={self.model.nv}, "
            f"end-effector={ee_frame!r}"
        )
##Joint limits extraction
    def _joint_angle_limits(self) -> Tuple[Array, Array]:
        """Extract six physical angle limits, with a conservative fallback."""
        lower = np.full(6, -2.0 * np.pi)
        upper = np.full(6, +2.0 * np.pi)
        for i, jid in enumerate(self.joint_ids):
            joint = self.model.joints[jid]
            if joint.nq == 1:
                lo = float(self.model.lowerPositionLimit[joint.idx_q])
                hi = float(self.model.upperPositionLimit[joint.idx_q])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    lower[i], upper[i] = lo, hi
            elif joint.nq != 2:
                raise ValueError(
                    f"Unsupported configuration size nq={joint.nq} for "
                    f"{self.joint_names[i]!r}"
                )
        return lower, upper
###Convert controller angles to Pinnocchio configuration
    def theta_to_q(self, theta: Array) -> Array:
        theta = np.asarray(theta, dtype=float).reshape(6)
        q = self.q_neutral.copy()
        for angle, jid in zip(theta, self.joint_ids):
            joint = self.model.joints[jid]
            if joint.nq == 1:
                
                q[joint.idx_q] = angle
            elif joint.nq == 2:
                # Pinocchio unbounded revolute configuration representation.
                q[joint.idx_q : joint.idx_q + 2] = [np.cos(angle), np.sin(angle)]
            else:  # Guarded in __init__; retained for defensive programming.
                raise RuntimeError("Unsupported UR joint representation")
        return q
####Forward kinematics and Jacobian
    def pose_and_jacobian(self, theta: Array) -> Tuple[pin.SE3, Array]:
        q = self.theta_to_q(theta)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id].copy() ##placement.translation, placement.rotation
        J_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return placement, J_full[:, self.velocity_indices] ##- the current tool pose; a 6×6 UR3e Jacobian.

    def pose(self, theta: Array) -> pin.SE3:
        return self.pose_and_jacobian(theta)[0] ###only the current tool pose, without the Jacobian.

    @staticmethod
    def pose_error(current: pin.SE3, desired: pin.SE3) -> Array:
        """World-aligned error [position; rotation-vector], desired-current."""
        error = np.empty(6)
        error[:3] = desired.translation - current.translation ##pose error in translation
        # R_des R_cur^T maps current world-aligned orientation to desired.
        error[3:] = pin.log3(desired.rotation @ current.rotation.T)
        return error
###final IK solver for the terminal pose
### This solves IK once for the final desired pose, using damped least squares and line search.
## It does not solve IK for every step in the trajectory,
# which is handled by the MPC controller.
    def solve_terminal_ik(
        self,
        desired: pin.SE3,
        seed: Array,
        max_iterations: int = 200,
        tolerance: float = 1.0e-6,
        damping: float = 1.0e-6,
    ) -> Array:
        """Solve IK once for P(T), using damped least squares and line search."""
        theta = np.clip(np.asarray(seed, dtype=float).copy(), self.lower, self.upper)
        for _ in range(max_iterations):
            current, J = self.pose_and_jacobian(theta)
            error = self.pose_error(current, desired)
            if np.linalg.norm(error) < tolerance:
                return theta

            step = J.T @ np.linalg.solve(
                J @ J.T + damping * np.eye(6), error
            )
            step = np.clip(step, -0.20, 0.20)

            old_norm = np.linalg.norm(error)
            accepted = False
            scale = 1.0
            for _ in range(10):
                candidate = np.clip(
                    theta + scale * step, self.lower, self.upper
                )
                candidate_error = self.pose_error(self.pose(candidate), desired)
                if np.linalg.norm(candidate_error) < old_norm:
                    theta = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break

        residual = np.linalg.norm(self.pose_error(self.pose(theta), desired))
        raise RuntimeError(f"Terminal IK did not converge; residual={residual:.3e}")

## Smooth SE(3) reference trajectory generation, 
# using a quintic polynomial for the position and rotation vector. 
# The twist is the time derivative of the pose, which is used as a reference for the MPC controller.
def make_se3_trajectory(
    start: pin.SE3,
    delta_position: Array,
    delta_rotation: Array,
    duration: float,
    dt: float,
) -> Tuple[Array, list[PoseSample], pin.SE3]:
    count = int(np.ceil(duration / dt)) + 1
    times = np.linspace(0.0, duration, count)
    s = times / duration
    sigma = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    sigma_dot = (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4) / duration #quintic polynomial derivative

    delta_position = np.asarray(delta_position, dtype=float).reshape(3)
    delta_rotation = np.asarray(delta_rotation, dtype=float).reshape(3)
    samples: list[PoseSample] = []
    for a, da in zip(sigma, sigma_dot):
        position = start.translation + a * delta_position
        rotation = pin.exp3(a * delta_rotation) @ start.rotation
        twist = np.concatenate((da * delta_position, da * delta_rotation))
        samples.append(PoseSample(position, rotation, twist))

    goal = pin.SE3(
        pin.exp3(delta_rotation) @ start.rotation,
        start.translation + delta_position,
    )
    return times, samples, goal

###Extract a reference MPC horizon of poses and twists from the trajectory samples.
def reference_horizon(
    samples: Sequence[PoseSample], start_index: int, horizon: int
) -> Tuple[list[pin.SE3], Array, bool]:
    indices = np.minimum(
        np.arange(start_index, start_index + horizon + 1), len(samples) - 1
    )
    poses = [
        pin.SE3(samples[i].rotation, samples[i].position) for i in indices
    ]
    twists = np.vstack([samples[i].twist for i in indices[:-1]])
    reaches_global_end = start_index + horizon >= len(samples) - 1
    return poses, twists, reaches_global_end

##MPC setup
class CartesianTwistMPC:
    """Sequentially linearized Cartesian MPC solved as one small OSQP QP."""

    def __init__(
        self,
        robot: UR3ePinocchio,
        horizon: int,
        dt: float,
        dq_max: Array,
        ddq_max: Array,
        collision: Optional[StaticBoxCollision] = None,
        collision_margin: float = 0.05,
    ) -> None:
        self.robot = robot
        self.N = int(horizon)
        self.dt = float(dt)
        self.dq_max = np.asarray(dq_max, dtype=float).reshape(6)
        self.ddq_max = np.asarray(ddq_max, dtype=float).reshape(6)
        self.collision = collision
        self.collision_margin = float(collision_margin)
        if self.collision_margin < 0.0:
            raise ValueError("collision_margin must be non-negative")
        self.last_min_clearance = np.inf
        self.last_active_collision_constraints = 0

        self.Q_twist = np.diag([30.0] * 3 + [12.0] * 3)
        self.Q_pose = np.diag([80.0] * 3 + [25.0] * 3)
        self.Q_terminal_pose = np.diag([300.0] * 3 + [100.0] * 3)
        self.Q_terminal_joint = 0.1 * np.eye(6)
        self.R_velocity = 2.0e-2 * np.eye(6)
        self.R_smooth = 8.0e-2 * np.eye(6)
        self.Kp_pose = np.diag([3.0] * 3 + [2.0] * 3)
        self.warm_start = np.zeros((self.N, 6))

        self._variable_count = 6 * self.N
        self._integration = self._make_integration_matrix()
        self._velocity_difference = self._make_velocity_difference_matrix()
        self._constraint_matrix = sparse.vstack(
            (
                sparse.eye(self._variable_count, format="csc"),
                self._integration,
                self._velocity_difference,
            ),
            format="csc",
        )
        self._p_rows, self._p_cols = self._upper_triangle_indices()
        self._solver: Optional[osqp.OSQP] = None

    def _make_integration_matrix(self) -> sparse.csc_matrix:
        """Map stacked velocities to theta[1:] under theta[k+1]=theta[k]+dt*dq[k]."""
        matrix = np.zeros((self._variable_count, self._variable_count))
        for state_step in range(self.N):
            for command_step in range(state_step + 1):
                row = slice(6 * state_step, 6 * (state_step + 1))
                column = slice(6 * command_step, 6 * (command_step + 1))
                matrix[row, column] = self.dt * np.eye(6)
        return sparse.csc_matrix(matrix)

    def _make_velocity_difference_matrix(self) -> sparse.csc_matrix:
        """Map stacked velocities to [dq[0], dq[1]-dq[0], ...]."""
        matrix = np.eye(self._variable_count)
        for step in range(1, self.N):
            row = slice(6 * step, 6 * (step + 1))
            column = slice(6 * (step - 1), 6 * step)
            matrix[row, column] = -np.eye(6)
        return sparse.csc_matrix(matrix)

    def _upper_triangle_indices(self) -> Tuple[Array, Array]:
        """Return OSQP's column-major ordering for a dense upper-triangular P."""
        rows = np.concatenate(
            [np.arange(column + 1) for column in range(self._variable_count)]
        )
        columns = np.concatenate(
            [np.full(column + 1, column) for column in range(self._variable_count)]
        )
        return rows, columns

    def _stack_pose_errors(
        self, current_pose: pin.SE3, pose_reference: Sequence[pin.SE3]
    ) -> Array:
        return np.concatenate(
            [self.robot.pose_error(current_pose, reference) for reference in pose_reference]
        )

    def _predict_theta(self, theta0: Array, dq: Array) -> Array:
        theta = np.empty((self.N + 1, 6))
        theta[0] = theta0
        for step in range(self.N):
            theta[step + 1] = theta[step] + self.dt * dq[step]
        return theta

    def current_clearance(self, theta: Array) -> float:
        """Minimum link/tool to obstacle clearance; infinity if disabled."""
        if self.collision is None:
            return np.inf
        return self.collision.minimum_distance(theta)

    def solve(
        self,
        theta0: Array,
        dq_previous: Array,
        pose_reference: Sequence[pin.SE3],
        twist_reference: Array,
        theta_terminal: Optional[Array] = None,
    ) -> Tuple[Array, Array, float]:
        if len(pose_reference) != self.N + 1:
            raise ValueError(f"pose_reference must contain {self.N + 1} poses")
        if twist_reference.shape != (self.N, 6):
            raise ValueError(
                f"twist_reference must have shape {(self.N, 6)}"
            )

        start = time.perf_counter()
        current_pose, J = self.robot.pose_and_jacobian(theta0)
        jacobian_horizon = sparse.kron(
            sparse.eye(self.N, format="csc"), sparse.csc_matrix(J), format="csc"
        )
        pose_mapping = jacobian_horizon @ self._integration

        # theta[k]-theta0 contains the first k commands; its first block is zero.
        state_mapping = sparse.vstack(
            (sparse.csc_matrix((6, self._variable_count)), pose_mapping[:-6]),
            format="csc",
        )
        kp_horizon = sparse.kron(
            sparse.eye(self.N, format="csc"), sparse.csc_matrix(self.Kp_pose), format="csc"
        )
        twist_mapping = jacobian_horizon + kp_horizon @ state_mapping

        # e_pose ~= target - pose_mapping @ U.  This retains the original
        # feedforward-plus-proportional Cartesian velocity objective.
        pose_target = self._stack_pose_errors(current_pose, pose_reference[1:])
        twist_target = (
            twist_reference.ravel()
            + kp_horizon @ self._stack_pose_errors(current_pose, pose_reference[:-1])
        )
        pose_weight = sparse.block_diag(
            [self.Q_pose] * (self.N - 1) + [self.Q_terminal_pose], format="csc"
        )
        twist_weight = sparse.kron(
            sparse.eye(self.N, format="csc"), sparse.csc_matrix(self.Q_twist), format="csc"
        )
        velocity_weight = sparse.kron(
            sparse.eye(self.N, format="csc"), sparse.csc_matrix(self.R_velocity), format="csc"
        )
        smoothness_weight = sparse.kron(
            sparse.eye(self.N, format="csc"), sparse.csc_matrix(self.R_smooth), format="csc"
        )
        #QP quadratic penalties objectives
        P = (
            twist_mapping.T @ twist_weight @ twist_mapping
            + pose_mapping.T @ pose_weight @ pose_mapping
            + velocity_weight
            + self._velocity_difference.T
            @ smoothness_weight
            @ self._velocity_difference
        ).toarray()
        linear = -np.asarray(
            twist_mapping.T @ twist_weight @ twist_target
            + pose_mapping.T @ pose_weight @ pose_target
        ).ravel()

        if theta_terminal is not None:
            terminal_mapping = self._integration[-6:]
            terminal_target = wrap_to_pi(theta_terminal - theta0)
            P += np.asarray(
                terminal_mapping.T @ self.Q_terminal_joint @ terminal_mapping
            )
            linear -= np.asarray(
                terminal_mapping.T @ self.Q_terminal_joint @ terminal_target
            ).ravel()

        # OSQP uses 1/2 U.T @ P @ U + linear.T @ U.
        P = 2.0 * P + 1.0e-9 * np.eye(self._variable_count)
        linear = 2.0 * linear
        velocity_offset = np.zeros(self._variable_count)
        velocity_offset[:6] = dq_previous
        allowed_change = np.tile(self.ddq_max * self.dt, self.N)
        lower_bound = np.concatenate(
            (
                -np.tile(self.dq_max, self.N),
                np.tile(self.robot.lower - theta0, self.N),
                velocity_offset - allowed_change,
            )
        )
        upper_bound = np.concatenate(
            (
                np.tile(self.dq_max, self.N),
                np.tile(self.robot.upper - theta0, self.N),
                velocity_offset + allowed_change,
            )
        )

        # Sequential convex collision constraints.  GJK is evaluated at the
        # previous MPC rollout, then d(q) is linearised using dd/dq = n.T @ Jp.
        collision_rows: list[Array] = []
        collision_upper_bounds: list[float] = []
        if self.collision is not None:
            nominal_dq = self.warm_start
            nominal_theta = self._predict_theta(theta0, nominal_dq)
            nominal_u = nominal_dq.ravel()
            self.last_min_clearance = self.current_clearance(theta0)
            for step in range(self.N):
                state_mapping = self._integration[6 * step : 6 * (step + 1)]
                nominal_displacement = state_mapping @ nominal_u
                for sample in self.collision.query(nominal_theta[step + 1]):
                    self.last_min_clearance = min(
                        self.last_min_clearance, sample.distance
                    )
                    # d + g(SU - SU_nom) >= margin
                    # becomes -gS U <= d - margin - gS U_nom.
                    collision_rows.append(
                        -np.asarray(sample.gradient @ state_mapping).ravel()
                    )
                    collision_upper_bounds.append(
                        sample.distance
                        - self.collision_margin
                        - float(sample.gradient @ nominal_displacement)
                    )
        self.last_active_collision_constraints = len(collision_rows)
        constraint_matrix = self._constraint_matrix
        if collision_rows:
            constraint_matrix = sparse.vstack(
                (constraint_matrix, sparse.csc_matrix(np.vstack(collision_rows))),
                format="csc",
            )
            lower_bound = np.concatenate(
                (lower_bound, np.full(len(collision_rows), -np.inf))
            )
            upper_bound = np.concatenate(
                (upper_bound, np.asarray(collision_upper_bounds))
            )

        p_values = P[self._p_rows, self._p_cols]
        # Collision rows change after every GJK linearisation, so OSQP must be
        # rebuilt with the new matrix.  The primal solution is still warm-started.
        if self._solver is None or collision_rows:
            P_sparse = sparse.csc_matrix(
                (p_values, (self._p_rows, self._p_cols)),
                shape=(self._variable_count, self._variable_count),
            )
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=P_sparse,
                q=linear,
                A=constraint_matrix,
                l=lower_bound,
                u=upper_bound,
                verbose=False,
                warm_starting=True,
                polishing=False,
                eps_abs=1.0e-5,
                eps_rel=1.0e-5,
                max_iter=400,
                time_limit=0.02,
            )
        else:
            self._solver.update(Px=p_values, q=linear, l=lower_bound, u=upper_bound)

        self._solver.warm_start(x=self.warm_start.ravel())
        result = self._solver.solve()
        solve_time = time.perf_counter() - start
        logger.info(f"solver status: {result.info.status}, solve time: {solve_time:.3e} s")
        logger.info(f"primal residual: {result.info.prim_res:.3e}, dual residual: {result.info.dual_res:.3e}")
        if result.info.status != "solved":
            raise RuntimeError(f"MPC failed: {result.info.status}")

        optimal = np.asarray(result.x).reshape(self.N, 6)
        self.warm_start[:-1] = optimal[1:]
        self.warm_start[-1] = optimal[-1]
        return optimal[0].copy(), optimal, solve_time

##Controller 
def run_controller(
    robot: UR3ePinocchio,
    mpc: CartesianTwistMPC,
    samples: Sequence[PoseSample],
    theta_terminal: Array,
    theta_initial: Array,
    hardware: bool,
    robot_ip: str,
    viewer: Optional[MeshcatSimulationViewer] = None,
) -> dict[str, Array]:
    """Run either a kinematic simulation or an explicitly enabled UR3e."""
    rtde_c = rtde_r = None
    if hardware:
        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "Hardware mode requires the ur_rtde Python package"
            ) from exc
        rtde_frequency = 1.0 / mpc.dt
        rtde_c = rtde_control.RTDEControlInterface(robot_ip, frequency=rtde_frequency)
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip, frequency=rtde_frequency)
        theta = np.asarray(rtde_r.getActualQ(), dtype=float)
        dq_previous = np.asarray(rtde_r.getActualQd(), dtype=float) ##on hardware, we get the current joint angles and velocities from the robot
    else:
        theta = theta_initial.copy()
        dq_previous = np.zeros(6)

    theta_log, actual_pose_log, reference_pose_log = [], [], []
    command_log, solve_time_log, clearance_log = [], [], []
    if viewer is not None:
        viewer.display(theta)

    try:
        for step in range(len(samples)):
            cycle_start = rtde_c.initPeriod() if hardware else None
            if hardware:
                theta = np.asarray(rtde_r.getActualQ(), dtype=float)
                dq_previous = np.asarray(rtde_r.getActualQd(), dtype=float)

            pose_ref, twist_ref, at_end = reference_horizon(
                samples, step, mpc.N
            )
            dq_command, _, solve_time = mpc.solve(
                theta,
                dq_previous,
                pose_ref,
                twist_ref,
                theta_terminal if at_end else None,
            )

            if hardware:
                # speedJ(qd, acceleration, time).  The explicit time prevents a
                # stale velocity command from being held indefinitely.
                # speedJ accepts one acceleration for all joints.  The CLI uses
                # equal limits; min() is conservative if this class is reused
                # with unequal per-joint limits.
                rtde_c.speedJ(
                    dq_command.tolist(), float(np.min(mpc.ddq_max)), mpc.dt
                )
                rtde_c.waitPeriod(cycle_start)
            else:
                theta = theta + mpc.dt * dq_command

            actual = robot.pose(theta)
            theta_log.append(theta.copy())
            actual_pose_log.append(
                np.concatenate(
                    (actual.translation, pin.log3(actual.rotation))
                )
            )
            desired = samples[step]
            reference_pose_log.append(
                np.concatenate(
                    (desired.position, pin.log3(desired.rotation))
                )
            )
            command_log.append(dq_command)
            solve_time_log.append(solve_time)
            clearance_log.append(mpc.current_clearance(theta))
            dq_previous = dq_command
            if viewer is not None:
                viewer.display(theta)
                # Simulation otherwise finishes as fast as the QP solves, which
                # is too quick to inspect interactively in the browser.
                time.sleep(mpc.dt)

            if solve_time > mpc.dt:
                print(
                    f"Warning: MPC solve {1e3*solve_time:.1f} ms exceeds "
                    f"dt={1e3*mpc.dt:.1f} ms at step {step}"
                )
    finally:
        if hardware and rtde_c is not None:
            rtde_c.speedStop(1.0)
            rtde_c.stopScript()

    return {
        "theta": np.asarray(theta_log),
        "actual_pose": np.asarray(actual_pose_log),
        "reference_pose": np.asarray(reference_pose_log),
        "dq": np.asarray(command_log),
        "solve_time": np.asarray(solve_time_log),
        "clearance": np.asarray(clearance_log),
    }


##Plotting and command-line interface

def plot_results(times: Array, result: dict[str, Array]) -> None:
    import matplotlib.pyplot as plt

    count = len(result["actual_pose"])
    t = times[:count]
    position_error = np.linalg.norm(
        result["reference_pose"][:, :3] - result["actual_pose"][:, :3], axis=1
    )

    show_clearance = np.isfinite(result["clearance"]).any()
    figure_rows = 4 if show_clearance else 3
    fig, axes = plt.subplots(figure_rows, 1, figsize=(10, 3 * figure_rows), sharex=True)
    axes[0].plot(t, result["reference_pose"][:, :3], "--")
    axes[0].plot(t, result["actual_pose"][:, :3])
    axes[0].set_ylabel("tool position [m]")
    axes[0].grid(True)

    axes[1].plot(t, 1000.0 * position_error)
    axes[1].set_ylabel("position error [mm]")
    axes[1].grid(True)

    axes[2].plot(t, result["dq"])
    axes[2].set_ylabel("joint velocity [rad/s]")
    axes[2].grid(True)

    if show_clearance:
        axes[3].plot(t, 1000.0 * result["clearance"])
        axes[3].set_ylabel("min clearance [mm]")
        axes[3].grid(True)
        axes[3].set_xlabel("time [s]")
    else:
        axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        default="/home/ibrahim/Documents/PhD/PD_Projects/SDCOL-main-complete/data/urdf/ur3e/ur3e_fixed.urdf",
    )
    parser.add_argument("--ee-frame", default="tool0")
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--q0-deg",
        nargs=6,
        type=float,
        default=[0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
    )
    parser.add_argument(
        "--delta-position",
        nargs=3,
        type=float,
        default=[0.11, 0.00, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="base/world-frame translation in metres",
    )
    parser.add_argument(
        "--delta-rotation",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="base/world-frame rotation vector in radians",
    )
    parser.add_argument("--max-joint-speed", type=float, default=0.6)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.5)
    parser.add_argument(
        "--obstacle-center",
        nargs=3,
        type=float,
        default=None,
        metavar=("OX", "OY", "OZ"),
        help="enable one static world-frame box obstacle in simulation",
    )
    parser.add_argument(
        "--obstacle-size",
        nargs=3,
        type=float,
        default=[0.08, 0.08, 0.08],
        metavar=("SX", "SY", "SZ"),
        help="static-box full X/Y/Z side lengths in metres",
    )
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.05,
        help="minimum link/tool surface clearance from the obstacle [m]",
    )
    parser.add_argument(
        "--tool-collision-radius",
        type=float,
        default=0.03,
        help="protective sphere radius centred at tool0 [m]",
    )
    parser.add_argument(
        "--check-collision-gradient",
        action="store_true",
        help="compare GJK witness-point gradients with finite differences at q0",
    )
    parser.add_argument(
        "--meshcat",
        action="store_true",
        help="show a live, real-time-paced Meshcat simulation viewer",
    )
    parser.add_argument(
        "--save-meshcat-gif",
        metavar="FILE.gif",
        help="save the Meshcat simulation trajectory as a looping GIF",
    )
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--confirm-hardware", action="store_true")
    parser.add_argument("--robot-ip", default="192.168.56.101")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hardware and not args.confirm_hardware:
        raise SystemExit(
            "Refusing to command hardware without --confirm-hardware. "
            "Run simulation first and verify the target and limits."
        )
    if args.hardware and args.obstacle_center is not None:
        raise SystemExit(
            "Static-obstacle collision MPC is simulation-only until its "
            "distance and gradient validation has been completed."
        )
    if args.hardware and args.meshcat:
        raise SystemExit("--meshcat is simulation-only; it does not visualise hardware.")
    if args.save_meshcat_gif is not None and not args.meshcat:
        raise SystemExit("--save-meshcat-gif requires --meshcat")
    if args.dt <= 0.0 or args.duration <= 0.0 or args.horizon < 1:
        raise ValueError("dt/duration must be positive and horizon >= 1")
    if (
        np.any(np.asarray(args.obstacle_size) <= 0.0)
        or args.tool_collision_radius <= 0.0
    ):
        raise ValueError("Obstacle side lengths and tool collision radius must be positive")
    if args.collision_margin < 0.0:
        raise ValueError("collision-margin must be non-negative")

    robot = UR3ePinocchio(args.urdf, args.ee_frame)
    theta_initial = np.deg2rad(np.asarray(args.q0_deg, dtype=float))
    if args.hardware:
        # Build the trajectory from the measured pose, not the CLI simulation q0.
        try:
            import rtde_receive
        except ImportError as exc:
            raise SystemExit("Hardware mode requires ur_rtde") from exc
        receiver = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        theta_initial = np.asarray(receiver.getActualQ(), dtype=float)
        del receiver

    start_pose = robot.pose(theta_initial)
    times, samples, goal_pose = make_se3_trajectory(
        start_pose,
        np.asarray(args.delta_position),
        np.asarray(args.delta_rotation),
        args.duration,
        args.dt,
    )

    # REVIEW: the only IK solve in the program is for the final Cartesian pose.
    theta_terminal = robot.solve_terminal_ik(goal_pose, theta_initial)
    print("Terminal IK [deg]:", np.round(np.rad2deg(theta_terminal), 3))

    collision = None
    if args.obstacle_center is not None:
        collision = StaticBoxCollision(
            robot,
            args.urdf,
            np.asarray(args.obstacle_center, dtype=float),
            np.asarray(args.obstacle_size, dtype=float),
            args.tool_collision_radius,
        )
        print(
            "Static box collision checking: "
            f"center={np.round(collision.center, 4)} m, "
            f"size={np.round(collision.size, 4)} m, "
            f"margin={args.collision_margin:.3f} m"
        )
        if args.check_collision_gradient:
            errors = collision.gradient_check(theta_initial)
            print(
                "Maximum initial finite-difference gradient error: "
                f"{max(errors.values()):.3e} m/rad"
            )

    mpc = CartesianTwistMPC(
        robot=robot,
        horizon=args.horizon,
        dt=args.dt,
        dq_max=np.full(6, args.max_joint_speed),
        ddq_max=np.full(6, args.max_joint_acceleration),
        collision=collision,
        collision_margin=args.collision_margin,
    )
    viewer = (
        MeshcatSimulationViewer(robot, collision, goal_pose.translation)
        if args.meshcat
        else None
    )
    result = run_controller(
        robot,
        mpc,
        samples,
        theta_terminal,
        theta_initial,
        args.hardware,
        args.robot_ip,
        viewer,
    )
    if viewer is not None:
        viewer.play_trajectory(result["theta"], args.dt)
        print("Meshcat playback installed and looping in the browser.")
        if args.save_meshcat_gif is not None:
            gif_path = viewer.save_gif(
                result["theta"], args.dt, args.save_meshcat_gif
            )
            print(f"Saved Meshcat animation: {gif_path}")

    final_actual = robot.pose(result["theta"][-1])
    final_error = robot.pose_error(final_actual, goal_pose)
    print(f"Final position error: {1e3*np.linalg.norm(final_error[:3]):.3f} mm")
    print(f"Final rotation error: {np.linalg.norm(final_error[3:]):.6f} rad")
    print(
        f"Mean/max MPC solve time: {1e3*np.mean(result['solve_time']):.1f} / "
        f"{1e3*np.max(result['solve_time']):.1f} ms"
    )
    if collision is not None:
        minimum_clearance = np.min(result["clearance"])
        print(
            f"Minimum link/tool clearance: {1e3*minimum_clearance:.3f} mm "
            f"(required: {1e3*args.collision_margin:.3f} mm)"
        )
    if not args.no_plot:
        plot_results(times, result)


if __name__ == "__main__":
    main()
