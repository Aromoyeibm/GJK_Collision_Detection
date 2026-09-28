"""
Point A to Point B trajectory for ONE 6-DoF arm (UR3), using Pinocchio.

Quick ref:
  1. Load the arm model (FK, Jacobians, IK).
  2. Define A and B as two END-EFFECTOR POSES.
  3. Linearly interpolate the pose A--->B  to a straight line in Cartesian space.
  4. IK at each waypoint (FK + Jacobian) -> joint traj q(t).

  >>> MPC + DCOL collision layer plugs in AFTER this, deforming q(t). <<<

"""
import os
import numpy as np
import pinocchio as pin
#from example_robot_data import load
import coal

#1. load the model (robot arm)
def load_ur3_model():
    from example_robot_data import load
    return load("ur3")

robot = load_ur3_model()
model = robot.model
collision_model = robot.collision_model
visual_model = robot.visual_model

data = model.createData()
collision_data = collision_model.createData()
EE = model.getFrameId("tool0") if model.existFrame("tool0") else model.getFrameId("ee_link")
#print([frame.name for frame in model.frames]) #to see all frames name
print(f"Loaded UR3: {model.nv} DoF", [model.names[i] for i in range(1, len(model.names))])

def fk_pose(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[EE].copy() #x,y,z, R

# COAL stores a box by its half-extents, so this is a 15 cm cube.
OBSTACLE_HALF_SIZE = 0.075
obstacle_geometry = coal.Box(
    OBSTACLE_HALF_SIZE,
    OBSTACLE_HALF_SIZE,
    OBSTACLE_HALF_SIZE,
)
#2. A and B as EE poses, 2 starting configurations

dA = np.array([ 0.0, -0.6,  0.8, -0.3,  1.2,  0.0])[:model.nv]
dB = np.array([-1.3, -1.2,  1.4, -0.6,  1.0,  0.0])[:model.nv]
qA_seed = pin.integrate(model, pin.neutral(model), dA)
qB_seed = pin.integrate(model, pin.neutral(model), dB)
poseA, poseB = fk_pose(qA_seed), fk_pose(qB_seed)
print("A (EE xyz):", np.round(poseA.translation, 3))
print("B (EE xyz):", np.round(poseB.translation, 3))

#3. linear interpolation of the POSE (Cartesian straight line)
# Also computes the rotation using SLERP (spherical linear interpolation) between the two quaternions.
def interp_pose(A, B, u):
    t = (1 - u) * A.translation + u * B.translation
    R = pin.Quaternion(pin.Quaternion(A.rotation).slerp(u, pin.Quaternion(B.rotation))).matrix()
    return pin.SE3(R, t)

#4. IK per waypoint (damped least squares)
def ik(target, q_init, iters=300, eps=1e-5, damp=1e-6, step=0.5):
    q = q_init.copy()
    err = np.zeros(6)
    for _ in range(iters):
        oMf = fk_pose(q)
        err = pin.log6(oMf.inverse() * target).vector          # Cartesian error (EE frame)
        if np.linalg.norm(err) < eps:
            break

        J = pin.computeFrameJacobian(model, data, q, EE)       # joint to Cartesian velocity map
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = pin.integrate(model, q, step * dq)

    oMf = fk_pose(q)
    position_error = np.linalg.norm(oMf.translation - target.translation)
    rotation_error = np.linalg.norm(pin.log3(oMf.rotation.T @ target.rotation))

    #return q, np.linalg.norm(err)
    return q, position_error, rotation_error

N = 60
q_traj, ee_path, tgt_path = [], [], []
q, worst = qA_seed.copy(), 0.0
worst_position_error = 0.0
worst_rotation_error = 0.0
for k in range(N):
    u = k / (N - 1)
    target = interp_pose(poseA, poseB, u)
    #q, e = ik(target, q)                                       # warm-start from previous q
    #worst = max(worst, e)
    q, position_error, rotation_error = ik(target, q)
    worst_position_error = max(worst_position_error, position_error)
    worst_rotation_error = max(worst_rotation_error, rotation_error)
    q_traj.append(q.copy()); ee_path.append(fk_pose(q).translation.copy())
    tgt_path.append(target.translation.copy())
q_traj, ee_path, tgt_path = map(np.array, (q_traj, ee_path, tgt_path))

# Obstacle motion in the world frame. Each position is the box centre.
# These bounds cover the current UR3 trajectory and leave room for motion.
workspace_min = np.array([0.10, -0.70, 0.00])
workspace_max = np.array([0.90, 0.45, 0.70])
obstacle_positions = np.empty((N, 3))
rng = np.random.default_rng(7)  # fixed seed makes the example repeatable
obstacle_positions[0] = rng.uniform(workspace_min, workspace_max)

for k in range(1, N):
    random_step = rng.normal(0.0, 0.035, size=3)
    next_position = obstacle_positions[k - 1] + random_step

    # Reflect at workspace boundaries instead of allowing the obstacle to leave.
    for axis in range(3):
        if next_position[axis] < workspace_min[axis]:
            next_position[axis] = workspace_min[axis] + (
                workspace_min[axis] - next_position[axis]
            )
        elif next_position[axis] > workspace_max[axis]:
            next_position[axis] = workspace_max[axis] - (
                next_position[axis] - workspace_max[axis]
            )
    obstacle_positions[k] = next_position

print("obstacle start (m):", np.round(obstacle_positions[0], 3))
print("obstacle end (m):  ", np.round(obstacle_positions[-1], 3))
#print(f"waypoints: {N}   worst IK residual: {worst:.2e}")
print(
    f"waypoints: {N}   "
    f"worst position error: {worst_position_error:.2e} m   "
    f"worst orientation error: {worst_rotation_error:.2e} rad"
)

# ---------- visualise ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def arm_points(q):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    pts = [data.oMi[i].translation.copy() for i in range(1, model.njoints)]
    pts.append(data.oMf[EE].translation.copy())
    return np.array(pts)

def draw_obstacle(center):
    half = OBSTACLE_HALF_SIZE
    corners = np.array([
        [center[0] + sx * half, center[1] + sy * half, center[2] + sz * half]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ])
    edges = (
        (0, 1), (0, 2), (0, 4), (1, 3),
        (1, 5), (2, 3), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7),
    )
    for start, end in edges:
        ax.plot(
            corners[[start, end], 0],
            corners[[start, end], 1],
            corners[[start, end], 2],
            color="crimson",
            lw=2,
        )
    ax.scatter(*center, color="crimson", s=35, label="moving obstacle")

fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
def draw(k):
    ax.cla()
    P = arm_points(q_traj[k])
    draw_obstacle(obstacle_positions[k])
    ax.plot(tgt_path[:,0], tgt_path[:,1], tgt_path[:,2], "--", color="#bbbbbb", lw=1.5, label="A to B target")
    ax.plot(ee_path[:k+1,0], ee_path[:k+1,1], ee_path[:k+1,2], color="#e07b39", lw=2, label="EE achieved")
    ax.plot(P[:,0], P[:,1], P[:,2], "-o", color="#2b6cb0", lw=4, ms=6)
    ax.scatter(*tgt_path[0], color="green", s=60); ax.text(*tgt_path[0], "  A", color="green")
    ax.scatter(*tgt_path[-1], color="red", s=60);  ax.text(*tgt_path[-1], " B", color="red")
    ax.scatter(0,0,0, color="k", s=40)
    ax.set_xlim(-0.6,0.9); ax.set_ylim(-0.7,0.5); ax.set_zlim(-0.1,0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"UR3  A to B   step {k+1}/{len(q_traj)}")
    ax.legend(loc="upper left", fontsize=8); ax.view_init(elev=22, azim=-60)

try:
    FuncAnimation(fig, draw, frames=len(q_traj), interval=60).save(
        "ab_trajectory.gif", writer=PillowWriter(fps=18))
    print("saved:", os.path.join(os.getcwd(), "ab_trajectory.gif"))
except Exception as e:
    print("GIF skipped (pip install pillow to enable):", e)
plt.close(fig)

# fig, ax = plt.subplots(figsize=(8, 4.5))
# for j in range(6): ax.plot(q_traj[:, j], lw=2, label=f"j{j+1}")
# ax.set_xlabel("waypoint (A to B)"); ax.set_ylabel("joint angle [rad]")
# ax.set_title("Joint-space reference q(t) from A to B"); ax.legend(ncol=6, fontsize=8)
# ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("joint_reference.png", dpi=120); plt.close(fig)
# print("saved:", os.path.join(os.getcwd(), "joint_reference.png"))





####kinematics motion###

"""
Point A to Point B trajectory for ONE 6-DoF arm (UR3), using Pinocchio.

Quick ref:
  1. Load the arm model (FK, Jacobians, IK).
  2. Define A and B as two END-EFFECTOR POSES.
  3. Linearly interpolate the pose A--->B  to a straight line in Cartesian space.
  4. IK at each waypoint (FK + Jacobian) -> joint traj q(t).

  >>> MPC + DCOL collision layer plugs in AFTER this, deforming q(t). <<<

"""
import os
import numpy as np
import pinocchio as pin
#from example_robot_data import load
import coal

#1. load the model (robot arm)
def load_ur3_model():
    from example_robot_data import load
    return load("ur3")

robot = load_ur3_model()
model = robot.model
collision_model = robot.collision_model
visual_model = robot.visual_model

data = model.createData()
collision_data = collision_model.createData()
EE = model.getFrameId("tool0") if model.existFrame("tool0") else model.getFrameId("ee_link")
#print([frame.name for frame in model.frames]) #to see all frames name
print(f"Loaded UR3: {model.nv} DoF", [model.names[i] for i in range(1, len(model.names))])

def fk_pose(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[EE].copy() #x,y,z, R

#2. A and B as EE poses, 2 starting configurations

dA = np.array([ 0.0, -0.6,  0.8, -0.3,  1.2,  0.0])[:model.nv]
dB = np.array([-1.3, -1.2,  1.4, -0.6,  1.0,  0.0])[:model.nv]
qA_seed = pin.integrate(model, pin.neutral(model), dA)
qB_seed = pin.integrate(model, pin.neutral(model), dB)
poseA, poseB = fk_pose(qA_seed), fk_pose(qB_seed)
print("A (EE xyz):", np.round(poseA.translation, 3))
print("B (EE xyz):", np.round(poseB.translation, 3))

#3. linear interpolation of the POSE (Cartesian straight line)
# Also computes the rotation using SLERP (spherical linear interpolation) between the two quaternions.
def interp_pose(A, B, u):
    t = (1 - u) * A.translation + u * B.translation
    R = pin.Quaternion(pin.Quaternion(A.rotation).slerp(u, pin.Quaternion(B.rotation))).matrix()
    return pin.SE3(R, t)

#4. IK per waypoint (damped least squares)
def ik(target, q_init, iters=300, eps=1e-5, damp=1e-6, step=0.5):
    q = q_init.copy()
    err = np.zeros(6)
    for _ in range(iters):
        oMf = fk_pose(q)
        err = pin.log6(oMf.inverse() * target).vector          # Cartesian error (EE frame)
        if np.linalg.norm(err) < eps:
            break

        J = pin.computeFrameJacobian(model, data, q, EE)       # joint to Cartesian velocity map
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
        q = pin.integrate(model, q, step * dq)

    oMf = fk_pose(q)
    position_error = np.linalg.norm(oMf.translation - target.translation)
    rotation_error = np.linalg.norm(pin.log3(oMf.rotation.T @ target.rotation))

    #return q, np.linalg.norm(err)
    return q, position_error, rotation_error

N = 60
q_traj, ee_path, tgt_path = [], [], []
q, worst = qA_seed.copy(), 0.0
worst_position_error = 0.0
worst_rotation_error = 0.0
for k in range(N):
    u = k / (N - 1)
    target = interp_pose(poseA, poseB, u)
    #q, e = ik(target, q)                                       # warm-start from previous q
    #worst = max(worst, e)
    q, position_error, rotation_error = ik(target, q)
    worst_position_error = max(worst_position_error, position_error)
    worst_rotation_error = max(worst_rotation_error, rotation_error)
    q_traj.append(q.copy()); ee_path.append(fk_pose(q).translation.copy())
    tgt_path.append(target.translation.copy())
q_traj, ee_path, tgt_path = map(np.array, (q_traj, ee_path, tgt_path))
#print(f"waypoints: {N}   worst IK residual: {worst:.2e}")
print(
    f"waypoints: {N}   "
    f"worst position error: {worst_position_error:.2e} m   "
    f"worst orientation error: {worst_rotation_error:.2e} rad"
)

# ---------- visualise ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def arm_points(q):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    pts = [data.oMi[i].translation.copy() for i in range(1, model.njoints)]
    pts.append(data.oMf[EE].translation.copy())
    return np.array(pts)

fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
def draw(k):
    ax.cla()
    P = arm_points(q_traj[k])
    ax.plot(tgt_path[:,0], tgt_path[:,1], tgt_path[:,2], "--", color="#bbbbbb", lw=1.5, label="A to B target")
    ax.plot(ee_path[:k+1,0], ee_path[:k+1,1], ee_path[:k+1,2], color="#e07b39", lw=2, label="EE achieved")
    ax.plot(P[:,0], P[:,1], P[:,2], "-o", color="#2b6cb0", lw=4, ms=6)
    ax.scatter(*tgt_path[0], color="green", s=60); ax.text(*tgt_path[0], "  A", color="green")
    ax.scatter(*tgt_path[-1], color="red", s=60);  ax.text(*tgt_path[-1], " B", color="red")
    ax.scatter(0,0,0, color="k", s=40)
    ax.set_xlim(-0.6,0.9); ax.set_ylim(-0.7,0.5); ax.set_zlim(-0.1,0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"UR3  A to B   step {k+1}/{len(q_traj)}")
    ax.legend(loc="upper left", fontsize=8); ax.view_init(elev=22, azim=-60)

try:
    FuncAnimation(fig, draw, frames=len(q_traj), interval=60).save(
        "ab_trajectory.gif", writer=PillowWriter(fps=18))
    print("saved:", os.path.join(os.getcwd(), "ab_trajectory.gif"))
except Exception as e:
    print("GIF skipped (pip install pillow to enable):", e)
plt.close(fig)

# fig, ax = plt.subplots(figsize=(8, 4.5))
# for j in range(6): ax.plot(q_traj[:, j], lw=2, label=f"j{j+1}")
# ax.set_xlabel("waypoint (A to B)"); ax.set_ylabel("joint angle [rad]")
# ax.set_title("Joint-space reference q(t) from A to B"); ax.legend(ncol=6, fontsize=8)
# ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("joint_reference.png", dpi=120); plt.close(fig)
# print("saved:", os.path.join(os.getcwd(), "joint_reference.png"))







###Ur3e_cartesian_velocity_mpc.py

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

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
    def __init__(
        self,
        robot: UR3ePinocchio,
        horizon: int,
        dt: float,
        dq_max: Array,
        ddq_max: Array,
    ) -> None:
        self.robot = robot
        self.N = int(horizon)
        self.dt = float(dt)
        self.dq_max = np.asarray(dq_max, dtype=float).reshape(6)
        self.ddq_max = np.asarray(ddq_max, dtype=float).reshape(6)

        self.Q_twist = np.diag([30.0] * 3 + [12.0] * 3)
        self.Q_pose = np.diag([80.0] * 3 + [25.0] * 3)
        self.Q_terminal_pose = np.diag([300.0] * 3 + [100.0] * 3)
        self.Q_terminal_joint = 0.1 * np.eye(6)
        self.R_velocity = 2.0e-2 * np.eye(6)
        self.R_smooth = 8.0e-2 * np.eye(6)
        self.Kp_pose = np.diag([3.0] * 3 + [2.0] * 3)
        self.warm_start = np.zeros((self.N, 6)) ##stores the previous optimal solution.. 
        ##..for warm-starting the next optimization...(faster and more stable convergence)
##State prediction for the MPC horizon,
# given the current joint angles and a sequence of joint velocities.
    def _predict_theta(self, theta0: Array, dq: Array) -> Array:
        theta = np.empty((self.N + 1, 6))
        theta[0] = theta0
        for k in range(self.N):
            theta[k + 1] = theta[k] + self.dt * dq[k] ##uses kinematic model; no dynamics, torque,masses or inertia model is used
        return theta

## MPC objective
    def _objective(
        self,
        flat_dq: Array,
        theta0: Array,
        dq_previous: Array,
        pose_reference: Sequence[pin.SE3],
        twist_reference: Array,
        theta_terminal: Optional[Array],
    ) -> float:
        dq = flat_dq.reshape(self.N, 6)
        theta = self._predict_theta(theta0, dq)
        cost = 0.0

        for k in range(self.N):
            ## At each step in the horizon, calculate tool pose, Jacobian, and Cartesian pose error.
            current_pose, J = self.robot.pose_and_jacobian(theta[k])
            pose_error = self.robot.pose_error(current_pose, pose_reference[k])

            #differential kinematics: predicted twist = J * dq[k]
            predicted_twist = J @ dq[k]
            commanded_twist = twist_reference[k] + self.Kp_pose @ pose_error ##feedforward + feedback
            twist_error = predicted_twist - commanded_twist

            ## cost function: weighted sum of squared errors for twist, pose, joint velocity, and smoothness (change in velocity)
            delta_dq = dq[k] - (dq_previous if k == 0 else dq[k - 1])
            cost += twist_error @ self.Q_twist @ twist_error ##track desired cartesian velocity(twist) and pose error
            cost += pose_error @ self.Q_pose @ pose_error ##track desired cartesian pose
            cost += dq[k] @ self.R_velocity @ dq[k] ##penalize large joint velocities
            cost += delta_dq @ self.R_smooth @ delta_dq ##penalize large changes in joint velocities (smoothness)

        final_pose = self.robot.pose(theta[-1])
        terminal_error = self.robot.pose_error(final_pose, pose_reference[-1])
        cost += terminal_error @ self.Q_terminal_pose @ terminal_error

        if theta_terminal is not None:
            ##The terminal joint penalty is added only 
            # once the current horizon can see the global end of the trajectory.
            joint_error = wrap_to_pi(theta[-1] - theta_terminal)
            cost += joint_error @ self.Q_terminal_joint @ joint_error
        return float(cost)

    ##COnstraint functions
    def _joint_limit_margin(self, flat_dq: Array, theta0: Array) -> Array:
        dq = flat_dq.reshape(self.N, 6)
        theta = self._predict_theta(theta0, dq)[1:]
        return np.concatenate(
            ((theta - self.robot.lower).ravel(),
             (self.robot.upper - theta).ravel())
        )

    def _acceleration_margin(
        self, flat_dq: Array, dq_previous: Array
    ) -> Array:
        dq = flat_dq.reshape(self.N, 6)
        delta = np.empty_like(dq)
        delta[0] = dq[0] - dq_previous
        delta[1:] = dq[1:] - dq[:-1]
        allowed = self.ddq_max * self.dt
        return np.concatenate(
            ((delta + allowed).ravel(), (allowed - delta).ravel())
        ) ## -allowed <= delta <= allowed, so we return allowed - |delta| >= 0

    ##Solving MPC

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

        bounds = [
            (-self.dq_max[j], self.dq_max[j])
            for _ in range(self.N)
            for j in range(6)
        ]
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: self._joint_limit_margin(x, theta0),
            },
            {
                "type": "ineq",
                "fun": lambda x: self._acceleration_margin(x, dq_previous),
            },
        ]

        start = time.perf_counter()
        result = minimize(
            self._objective,
            self.warm_start.ravel(),
            args=(
                theta0,
                dq_previous,
                pose_reference,
                twist_reference,
                theta_terminal,
            ),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 60, "ftol": 1.0e-5, "disp": False},
        )
        solve_time = time.perf_counter() - start
        if not result.success:
            raise RuntimeError(f"MPC failed: {result.message}")

        optimal = result.x.reshape(self.N, 6)
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
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        theta = np.asarray(rtde_r.getActualQ(), dtype=float)
        dq_previous = np.asarray(rtde_r.getActualQd(), dtype=float) ##on hardware, we get the current joint angles and velocities from the robot
    else:
        theta = theta_initial.copy()
        dq_previous = np.zeros(6)

    theta_log, actual_pose_log, reference_pose_log = [], [], []
    command_log, solve_time_log = [], []  ##lists for later plots and summary statistics

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
                rtde_c.speedJ(dq_command.tolist(), 1.0, mpc.dt)
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
            dq_previous = dq_command

            if solve_time > mpc.dt:
                print(
                    f"Warning: MPC solve {1e3*solve_time:.1f} ms exceeds "
                    f"dt={1e3*mpc.dt:.1f} ms at step {step}"
                )
    finally:
        if hardware and rtde_c is not None:
            rtde_c.stopJ(1.0)
            rtde_c.stopScript()

    return {
        "theta": np.asarray(theta_log),
        "actual_pose": np.asarray(actual_pose_log),
        "reference_pose": np.asarray(reference_pose_log),
        "dq": np.asarray(command_log),
        "solve_time": np.asarray(solve_time_log),
    }


##Plotting and command-line interface

def plot_results(times: Array, result: dict[str, Array]) -> None:
    import matplotlib.pyplot as plt

    count = len(result["actual_pose"])
    t = times[:count]
    position_error = np.linalg.norm(
        result["reference_pose"][:, :3] - result["actual_pose"][:, :3], axis=1
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, result["reference_pose"][:, :3], "--")
    axes[0].plot(t, result["actual_pose"][:, :3])
    axes[0].set_ylabel("tool position [m]")
    axes[0].grid(True)

    axes[1].plot(t, 1000.0 * position_error)
    axes[1].set_ylabel("position error [mm]")
    axes[1].grid(True)

    axes[2].plot(t, result["dq"])
    axes[2].set_ylabel("joint velocity [rad/s]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        default="/home/ibrahim/Documents/PhD/PD_Projects/SDCOL-main-complete/data/urdf/ur3e/ur3e_fixed.urdf",
    )
    parser.add_argument("--ee-frame", default="tool0")
    parser.add_argument("--dt", type=float, default=1/500)
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
        default=[-0.1, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="base/world-frame translation in metres",
    )
    parser.add_argument(
        "--delta-rotation",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        metavar=("RX", "RY", "RZ"),
        help="base/world-frame rotation vector in radians",
    )
    parser.add_argument("--max-joint-speed", type=float, default=0.6)
    parser.add_argument("--max-joint-acceleration", type=float, default=1.5)
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
    if args.dt <= 0.0 or args.duration <= 0.0 or args.horizon < 1:
        raise ValueError("dt/duration must be positive and horizon >= 1")

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

    mpc = CartesianTwistMPC(
        robot=robot,
        horizon=args.horizon,
        dt=args.dt,
        dq_max=np.full(6, args.max_joint_speed),
        ddq_max=np.full(6, args.max_joint_acceleration),
    )
    result = run_controller(
        robot,
        mpc,
        samples,
        theta_terminal,
        theta_initial,
        args.hardware,
        args.robot_ip,
    )

    final_actual = robot.pose(result["theta"][-1])
    final_error = robot.pose_error(final_actual, goal_pose)
    print(f"Final position error: {1e3*np.linalg.norm(final_error[:3]):.3f} mm")
    print(f"Final rotation error: {np.linalg.norm(final_error[3:]):.6f} rad")
    print(
        f"Mean/max MPC solve time: {1e3*np.mean(result['solve_time']):.1f} / "
        f"{1e3*np.max(result['solve_time']):.1f} ms"
    )
    if not args.no_plot:
        plot_results(times, result)


if __name__ == "__main__":
    main()





###ur3e_cartesian_velocity_mpc.py using OSQP.. before GJK algorithm

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
    ) -> None:
        self.robot = robot
        self.N = int(horizon)
        self.dt = float(dt)
        self.dq_max = np.asarray(dq_max, dtype=float).reshape(6)
        self.ddq_max = np.asarray(ddq_max, dtype=float).reshape(6)

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

        p_values = P[self._p_rows, self._p_cols]
        if self._solver is None:
            P_sparse = sparse.csc_matrix(
                (p_values, (self._p_rows, self._p_cols)),
                shape=(self._variable_count, self._variable_count),
            )
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=P_sparse,
                q=linear,
                A=self._constraint_matrix,
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
    command_log, solve_time_log = [], []  ##lists for later plots and summary statistics

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
            dq_previous = dq_command

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
    }


##Plotting and command-line interface

def plot_results(times: Array, result: dict[str, Array]) -> None:
    import matplotlib.pyplot as plt

    count = len(result["actual_pose"])
    t = times[:count]
    position_error = np.linalg.norm(
        result["reference_pose"][:, :3] - result["actual_pose"][:, :3], axis=1
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, result["reference_pose"][:, :3], "--")
    axes[0].plot(t, result["actual_pose"][:, :3])
    axes[0].set_ylabel("tool position [m]")
    axes[0].grid(True)

    axes[1].plot(t, 1000.0 * position_error)
    axes[1].set_ylabel("position error [mm]")
    axes[1].grid(True)

    axes[2].plot(t, result["dq"])
    axes[2].set_ylabel("joint velocity [rad/s]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)
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
        default=[0.1, 0.0, 0.0],
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
    if args.dt <= 0.0 or args.duration <= 0.0 or args.horizon < 1:
        raise ValueError("dt/duration must be positive and horizon >= 1")

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

    mpc = CartesianTwistMPC(
        robot=robot,
        horizon=args.horizon,
        dt=args.dt,
        dq_max=np.full(6, args.max_joint_speed),
        ddq_max=np.full(6, args.max_joint_acceleration),
    )
    result = run_controller(
        robot,
        mpc,
        samples,
        theta_terminal,
        theta_initial,
        args.hardware,
        args.robot_ip,
    )

    final_actual = robot.pose(result["theta"][-1])
    final_error = robot.pose_error(final_actual, goal_pose)
    print(f"Final position error: {1e3*np.linalg.norm(final_error[:3]):.3f} mm")
    print(f"Final rotation error: {np.linalg.norm(final_error[3:]):.6f} rad")
    print(
        f"Mean/max MPC solve time: {1e3*np.mean(result['solve_time']):.1f} / "
        f"{1e3*np.max(result['solve_time']):.1f} ms"
    )
    if not args.no_plot:
        plot_results(times, result)


if __name__ == "__main__":
    main()
