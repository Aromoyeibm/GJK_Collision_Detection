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
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_control import RTDEControlInterface as RTDEControl
#from example_robot_data import load
import coal
from loguru import logger
from copy import deepcopy
from mesh import replace_mesh_paths
#1. load the model (robot arm)
def load_ur3_model():
    from example_robot_data import load
    return load("ur3")



urdf = os.path.join(os.path.dirname(__file__), "..", "SDCOL-main-complete/data/urdf/ur3e/ur3e_fixed.urdf")



mesh= os.path.join(os.path.dirname(__file__), "..", "SDCOL-main-complete/data/urdf/ur3e/meshes")

replace_mesh_paths(
    os.path.join(os.path.dirname(__file__), "..", "SDCOL-main-complete/data/urdf/ur3e/ur3e.urdf"),
    "ur_description", 
    os.path.join(os.path.dirname(__file__), "..", "SDCOL-main-complete/data/urdf/ur3e"),
    os.path.join(os.path.dirname(__file__), "..", "SDCOL-main-complete/data/urdf/ur3e/ur3e_fixed.urdf")
    ) #replace the package:// path with file:// path for meshes

robot = pin.buildModelFromUrdf(urdf)
model = robot
# collision_model = robot.collision_model
# visual_model = robot.visual_model
# collision_model = pin.buildModelsFromUrdf(urdf, mesh)
# visual_model = pin.buildVisualModel(robot, visual_model)
data = model.createData()
# collision_data = collision_model.createData()
EE = model.getFrameId("tool0") if model.existFrame("tool0") else model.getFrameId("ee_link")
#print([frame.name for frame in model.frames]) #to see all frames name
print(f"Loaded UR3: {model.nv} DoF", [model.names[i] for i in range(1, len(model.names))])

def fk_pose(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[EE].copy() #x,y,z, R

#2. A and B as EE poses, 2 starting configurations

# dA = np.array([ 0.0, -0.6,  0.8, -0.3,  1.2,  0.0])[:model.nv]
# dB = np.array([-1.3, -1.2,  1.4, -0.6,  1.0,  0.0])[:model.nv]
# qA_seed = pin.integrate(model, pin.neutral(model), dA)
# qB_seed = pin.integrate(model, pin.neutral(model), dB)
# poseA, poseB = fk_pose(qA_seed), fk_pose(qB_seed)
# print("A (EE xyz):", np.round(poseA.translation, 3))
# print("B (EE xyz):", np.round(poseB.translation, 3))

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


#####UR3e coming in########
rtde_r = RTDEReceive("192.168.56.101")
rtde_c = RTDEControl("192.168.56.101")

q_init = np.asarray(rtde_r.getActualQ())
q = deepcopy(q_init)
poseA_6D = np.asarray(rtde_r.getActualTCPPose()) # real TCP pose
poseA = pin.SE3(pin.utils.rpyToMatrix(poseA_6D[3:]), poseA_6D[:3]) # convert to SE3
poseB = pin.SE3(pin.utils.rpyToMatrix(poseA_6D[3:]), poseA_6D[:3] + np.array([0.0, -0.2, 0.0])) # move 20cm in -y direction
logger.info(f"UR3e initial joint angles: {q_init}")
logger.info(f"UR3e initial TCP pose: {poseA}")





N = 60
q_traj, ee_path, tgt_path = [], [], []
# q, worst = qA_seed.copy(), 0.0
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
logger.info("tgt_path: {} -> {}".format(tgt_path[0], tgt_path[-1]))
q_ref = q_traj.copy()
#print(f"waypoints: {N}   worst IK residual: {worst:.2e}")
print(
    f"waypoints: {N}   "
    f"worst position error: {worst_position_error:.2e} m   "
    f"worst orientation error: {worst_rotation_error:.2e} rad"
)

#5. MPC
# The controller optimises joint velocities while tracking q_ref.
# A future collision module can add constraints through the optional constraint hook.
MPC_HORIZON = 12
MPC_DT = 0.10 #step time
MPC_V_MAX = 1.6 #joint velocity limit
MPC_W_TRACK = 1.0 #track weight
MPC_W_TERMINAL = 20.0#terminal weight
MPC_W_CONTROL = 0.02 #control weight

class KinematicMPC:
    def __init__(self, horizon=MPC_HORIZON, dt=MPC_DT):
        import scipy.sparse as sp
        import osqp

        if model.nq != model.nv:
            raise ValueError("This first MPC version expects nq == nv.")

        self.horizon = horizon
        self.dt = dt
        self.osqp = osqp
        self.sp = sp
        self.dimension = horizon * model.nv
        self.lower = model.lowerPositionLimit
        self.upper = model.upperPositionLimit

        integration = np.zeros((self.dimension, self.dimension))
        for step_index in range(1, horizon + 1):
            for input_index in range(step_index):
                row = slice((step_index - 1) * model.nv, step_index * model.nv)
                col = slice(input_index * model.nv, (input_index + 1) * model.nv)
                integration[row, col] = np.eye(model.nv)
        self.integration = dt * integration

        weights = np.concatenate([
            np.full((horizon - 1) * model.nv, MPC_W_TRACK),
            np.full(model.nv, MPC_W_TERMINAL),
        ])
        self.weight = np.diag(weights)
        P = 2 * (
            self.integration.T @ self.weight @ self.integration
            + MPC_W_CONTROL * np.eye(self.dimension)
        ) #U^T * P * U + q^T * U
        self.P = sp.csc_matrix((P + P.T) / 2)

    def rollout(self, q0, controls):
        configurations = []
        displacement = np.zeros(model.nv)
        for step_index in range(self.horizon):
            displacement += self.dt * controls[
                step_index * model.nv:(step_index + 1) * model.nv
            ]
            configurations.append(pin.integrate(model, q0, displacement))
        return configurations

    def solve(self, q0, reference_index):
        target_displacement = np.zeros(self.dimension)
        for step_index in range(1, self.horizon + 1):
            target = q_ref[min(reference_index + step_index, len(q_ref) - 1)]
            target_displacement[(step_index - 1) * model.nv:step_index * model.nv] = pin.difference(
                model, q0, target
            )

        linear_cost = -2 * self.integration.T @ self.weight @ target_displacement
        zero_controls = np.zeros(self.dimension)
        nominal_configurations = self.rollout(q0, zero_controls)

        rows = [np.eye(self.dimension)]
        lower = [-MPC_V_MAX * np.ones(self.dimension)]
        upper = [MPC_V_MAX * np.ones(self.dimension)]

        # Linearised joint-position limits:
        # q_nom + integration_k @ U must remain between lower and upper.
        for step_index, nominal_q in enumerate(nominal_configurations, start=1):
            integration_k = self.integration[
                (step_index - 1) * model.nv:step_index * model.nv,
                :,
            ]
            rows.append(integration_k)
            lower.append(self.lower - nominal_q)
            upper.append(self.upper - nominal_q)

        A = self.sp.csc_matrix(np.vstack(rows))
        problem = self.osqp.OSQP()
        problem.setup(
            self.P,
            linear_cost,
            A,
            np.concatenate(lower),
            np.concatenate(upper),
            verbose=False,
            max_iter=4000,
        )
        result = problem.solve()
        if result.x is None or result.info.status_val not in (1, 2):
            raise RuntimeError(f"MPC QP failed: {result.info.status}")
        return result.x[:model.nv], result.info.status


mpc = KinematicMPC()
q_mpc = deepcopy(q_init)
mpc_traj = [] ##online MPC solution 
mpc_controls = []
mpc_statuses = []
for k in range(N):
    control, status = mpc.solve(q_mpc, k)
    q_mpc = pin.integrate(model, q_mpc, MPC_DT * control)
    mpc_traj.append(q_mpc.copy())
    mpc_controls.append(control.copy())
    mpc_statuses.append(status)

mpc_traj = np.array(mpc_traj)
mpc_controls = np.array(mpc_controls)
mpc_ee_path = np.array([fk_pose(q).translation.copy() for q in mpc_traj])
mpc_position_error = np.array([
    np.linalg.norm(fk_pose(q).translation - target)
    for q, target in zip(mpc_traj, tgt_path)
])
print(
    f"MPC: {len(mpc_traj)} steps, status={mpc_statuses[-1]}, "
    f"final EE position error={mpc_position_error[-1]:.3e} m"
)

# Use the MPC trajectory for the animation below; q_ref remains the target.
q_traj = mpc_traj
ee_path = mpc_ee_path
logger.info(f"MPC trajectory: {ee_path[0]} -> {ee_path[-1]}")

# rtde_c.moveJ(q_traj[0].tolist(), 0.5, 0.5)
# i = 0
# try:
#     while True:
#         t_start = rtde_r.getActualTCPPose()
#         rtde_c.servoJ(q_traj[i].tolist(), 0.5, 0.5, 1.0/500, 0.1, 800)
#         rtde_c.waitForController(t_start)
#         if i < len(q_traj) - 1:
#             i += 1
#         else:
#             break
# except KeyboardInterrupt:
#     print("Interrupted by user.")
   



#For collision avoidance;
#distance d
#distance gradient ∂d/∂q


#visualise 
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