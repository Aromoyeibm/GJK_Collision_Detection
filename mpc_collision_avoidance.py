"""
================================================================================
 Online MPC collision-avoidance HARNESS for a 6-DoF arm (UR3), using Pinocchio.
================================================================================

WHAT THIS IS
  A test environment for a collision-avoidance *algorithm*. The arm tracks a
  point-A -> point-B reference while a dynamic obstacle moves randomly across
  its workspace. A receding-horizon MPC re-optimises the motion every tick and
  keeps the arm clear of the obstacle.

  The collision algorithm is a PLUG-IN. The MPC never calls GJK / DCOL / iDCOL
  directly -- it only calls a CollisionModule that returns, for the current
  configuration q and obstacle state:
        a signed distance d  (>0 = clear)   and   its gradient  dd/dq.
  Swap the algorithm by changing ONE line (CHECKER below). Three modules ship:
        SphereSetCollision  - analytic distance + exact gradient (default, fast)
        CoalGJKCollision    - true mesh distance via Coal/HPP-FCL (GJK baseline)
        DCOLCollision       - stub: drop your team's algorithm in here
  Your algorithm's job is exactly the CollisionModule.query() contract.

PIPELINE
  A->B reference (interp + IK)  ->  MPC (SCP-QP, OSQP)  ->  arm dodges obstacle
                                         ^
                                         |__ collision constraint d(q) >= margin
                                             supplied by the CollisionModule

RUN
  conda install -c conda-forge pinocchio example-robot-data osqp matplotlib
  python mpc_collision_avoidance.py
OUTPUT
  mpc_avoidance.gif   (arm + moving obstacle)
  mpc_metrics.png     (min clearance & solve time over the run)

Author scaffold: Ibrahim (Akinjobi Aromoye). Collision algorithm slots into
CollisionModule -- see DCOLCollision.
"""
import os, time
from abc import ABC, abstractmethod
import numpy as np
import pinocchio as pin

# ------------------------------------------------------------------ CONFIG ----
CHECKER   = "sphere"     # "sphere" | "gjk" | "dcol"  <-- swap the algorithm here
HORIZON   = 12           # MPC prediction horizon (steps)
DT        = 0.10         # control period [s]
V_MAX     = 1.6          # joint speed limit [rad/s]
MARGIN    = 0.05         # required clearance [m] (absorbs linearisation error)
LINK_R    = 0.06         # arm "thickness" (sphere-swept radius) [m]
OBST_R    = 0.09         # obstacle radius [m]
N_REF     = 60           # reference waypoints A->B
SIM_STEPS = 180          # simulation ticks
REF_SPAN  = 110          # ticks to traverse A->B (rest = settle/recover)
SEED      = 6

# =============================================================== 1. ROBOT =====
def load_ur3():
    """Return (model, collision_model). Portable across install routes."""
    env = os.environ.get("UR3_URDF", "")
    if env:
        m = pin.buildModelFromUrdf(env)
        return m, None
    try:
        from example_robot_data import load
        r = load("ur3"); return r.model, r.collision_model
    except Exception:
        pass
    try:  # pip install robot_descriptions xacrodoc
        from robot_descriptions.loaders.pinocchio import load_robot_description
        r = load_robot_description("ur3e_description"); return r.model, r.collision_model
    except Exception:
        pass
    raise SystemExit("Could not load UR3. Try: conda install -c conda-forge example-robot-data")

model, collision_model = load_ur3()
data = model.createData()
nv   = model.nv
EE   = model.getFrameId("tool0") if model.existFrame("tool0") else model.getFrameId("ee_link")
print(f"Loaded UR3: {nv} DoF ->", [model.names[i] for i in range(1, len(model.names))])

def fk_pose(q):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    return data.oMf[EE].copy()

# Arm as a set of control points (joint origins + tool). The MPC keeps every
# one of these clear of the obstacle. Sampling more points = finer body model.
CP_JOINTS = list(range(2, model.njoints))          # skip world/base
def arm_control_points(q):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    pts = [data.oMi[j].translation.copy() for j in CP_JOINTS]
    pts.append(data.oMf[EE].translation.copy())
    return pts
def arm_point_jacobians(q):
    """Translational Jacobian (3 x nv) of each control point, world-aligned."""
    pin.computeJointJacobians(model, data, q); pin.updateFramePlacements(model, data)
    Js = [pin.getJointJacobian(model, data, j, pin.LOCAL_WORLD_ALIGNED)[:3].copy() for j in CP_JOINTS]
    Js.append(pin.getFrameJacobian(model, data, EE, pin.LOCAL_WORLD_ALIGNED)[:3].copy())
    return Js

# ====================================================== 2. A->B REFERENCE =====
def interp_pose(A, B, u):
    t = (1 - u) * A.translation + u * B.translation
    R = pin.Quaternion(pin.Quaternion(A.rotation).slerp(u, pin.Quaternion(B.rotation))).matrix()
    return pin.SE3(R, t)

def ik(target, q_init, iters=200, eps=1e-5, damp=1e-6, step=0.5):
    q = q_init.copy()
    for _ in range(iters):
        oMf = fk_pose(q)
        e = pin.log6(oMf.inverse() * target).vector
        if np.linalg.norm(e) < eps: break
        J = pin.computeFrameJacobian(model, data, q, EE)
        q = pin.integrate(model, q, step * (J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)))
    return q

# A and B as reachable EE poses (from two joint seeds; nv-safe for UR3/UR3e)
dA = np.array([ 0.0, -0.6, 0.8, -0.3, 1.2, 0.0])[:nv]
dB = np.array([-1.3, -1.2, 1.4, -0.6, 1.0, 0.0])[:nv]
qA = pin.integrate(model, pin.neutral(model), dA)
qB = pin.integrate(model, pin.neutral(model), dB)
poseA, poseB = fk_pose(qA), fk_pose(qB)

q_ref, q = [], qA.copy()
for k in range(N_REF):
    q = ik(interp_pose(poseA, poseB, k / (N_REF - 1)), q)   # warm-started IK
    q_ref.append(q.copy())
q_ref = np.array(q_ref)
print("A (EE xyz):", np.round(poseA.translation, 3), " B:", np.round(poseB.translation, 3))

# ================================================ 3. COLLISION SOCKET ==========
class CollisionModule(ABC):
    """
    THE PLUG-IN CONTRACT.  Implement query() for GJK / DCOL / iDCOL / your algo.

        query(q, obstacle) -> list of (d_i, Jd_i)
            d_i  : signed distance [m] between arm feature i and the obstacle
                   (>0 clear, <0 penetrating)
            Jd_i : gradient d(d_i)/dq, shape (nv,)   <-- this is what the MPC needs

    The MPC turns each pair into a linearised constraint  d_i + Jd_i . dq >= margin.
    A differentiable algorithm (DCOL/iDCOL) supplies Jd_i analytically; a
    non-differentiable one (raw GJK) needs witness points or finite differences.
    """
    @abstractmethod
    def query(self, q, obstacle): ...

    def finite_diff_check(self, q, obstacle, h=1e-6):
        """Validate a module's analytic gradient against finite differences.
        Handy when your team wires in a new algorithm."""
        base = [d for d, _ in self.query(q, obstacle)]
        num = np.zeros((len(base), nv))
        for j in range(nv):
            dq = np.zeros(nv); dq[j] = h
            dp = [d for d, _ in self.query(pin.integrate(model, q, dq), obstacle)]
            dm = [d for d, _ in self.query(pin.integrate(model, q, -dq), obstacle)]
            num[:, j] = (np.array(dp) - np.array(dm)) / (2 * h)
        ana = np.array([Jd for _, Jd in self.query(q, obstacle)])
        return np.max(np.abs(ana - num))


class SphereSetCollision(CollisionModule):
    """Arm = spheres on control points, obstacle = sphere. Exact analytic gradient.
    Fast and dependency-free -> the default so the harness always runs."""
    def __init__(self, link_r=LINK_R):
        self.link_r = link_r
    def query(self, q, obstacle):
        c, r = obstacle.center, obstacle.radius
        out = []
        for p, J in zip(arm_control_points(q), arm_point_jacobians(q)):
            delta = p - c; dist = np.linalg.norm(delta)
            n = delta / max(dist, 1e-9)
            out.append((dist - r - self.link_r, n @ J))     # d,  dd/dq = n^T J_point
        return out


class CoalGJKCollision(CollisionModule):
    """True mesh distance via Coal / HPP-FCL (GJK/EPA). Gradient from witness
    points: dd/dq = n^T J_point(witness on link).  This is the 'GJK baseline'
    your algorithm competes against."""
    def __init__(self):
        if collision_model is None:
            raise RuntimeError("No collision_model available for the GJK module.")
        import coal
        self.cm = collision_model
        self.obst_gid = self.cm.addGeometryObject(
            pin.GeometryObject("mpc_obstacle", 0, pin.SE3.Identity(), coal.Sphere(OBST_R)))
        self.obstacle_pair_ids = []
        for i in range(self.cm.ngeoms):
            if i == self.obst_gid: continue
            self.cm.addCollisionPair(pin.CollisionPair(i, self.obst_gid))
            self.obstacle_pair_ids.append(len(self.cm.collisionPairs) - 1)
        self.cd = self.cm.createData()
    def query(self, q, obstacle):
        self.cm.geometryObjects[self.obst_gid].placement = pin.SE3(np.eye(3), obstacle.center)
        pin.computeJointJacobians(model, data, q)
        pin.updateGeometryPlacements(model, data, self.cm, self.cd, q)
        pin.computeDistances(model, data, self.cm, self.cd, q)
        out = []
        for res_idx in self.obstacle_pair_ids:
            dr = self.cd.distanceResults[res_idx]
            d = dr.min_distance
            p1 = np.array(dr.getNearestPoint1())            # witness on robot link
            p2 = np.array(dr.getNearestPoint2())            # witness on obstacle
            gid = self.cm.collisionPairs[res_idx].first
            jid = self.cm.geometryObjects[gid].parentJoint
            o   = data.oMi[jid].translation
            Jj  = pin.getJointJacobian(model, data, jid, pin.LOCAL_WORLD_ALIGNED)
            Jp  = Jj[:3] - pin.skew(p1 - o) @ Jj[3:]        # Jacobian of the witness point
            n   = p1 - p2; n /= max(np.linalg.norm(n), 1e-9)
            out.append((d - self.link_r_pad(), n @ Jp))
        return out
    def link_r_pad(self):  # meshes already have thickness; small pad only
        return 0.0


class DCOLCollision(CollisionModule):
    """
    >>> PLUG YOUR TEAM'S ALGORITHM IN HERE. <<<

    Represent each arm link and the obstacle as convex primitives (DCOL:
    polytope/capsule/cylinder/cone/ellipsoid/padded-polygon). For each link,
    call your algorithm to get a collision metric and its gradient, then return
    them as (d_i, Jd_i) in the SAME units/convention as the other modules:

        d_i  : signed clearance [m]  (or map your scaling alpha -> distance)
        Jd_i : d(d_i)/dq            (chain your dmetric/d(primitive pose) through
                                     the forward-kinematics Jacobian of the link)

    Everything downstream (MPC, metrics, plots) then works unchanged, so you can
    A/B this against SphereSet and CoalGJK in an identical harness.
    """
    def query(self, q, obstacle):
        raise NotImplementedError(
            "DCOLCollision.query() is a stub -- return [(d_i, dd_i/dq), ...] from your algorithm.")


def make_checker(name):
    if name == "sphere": return SphereSetCollision()
    if name == "gjk":    return CoalGJKCollision()
    if name == "dcol":   return DCOLCollision()
    raise ValueError(name)

# ================================================= 4. DYNAMIC OBSTACLE =========
class DynamicObstacle:
    """A sphere doing a smooth random walk inside a workspace box, seeded so runs
    are reproducible. Exposes .center, .radius, .velocity for prediction."""
    def __init__(self, center, radius=OBST_R, box_half=0.35, speed=0.14, seed=SEED):
        self.center = np.array(center, float)
        self.home   = self.center.copy()
        self.radius = radius
        self.box_half = box_half
        self.speed = speed
        self.rng = np.random.default_rng(seed)
        v = self.rng.normal(size=3); self.velocity = speed * v / np.linalg.norm(v)
    def step(self, dt):
        # smooth random heading change, renormalised to ~constant speed
        self.velocity += 0.05 * self.speed * self.rng.normal(size=3)
        self.velocity *= self.speed / max(np.linalg.norm(self.velocity), 1e-9)
        self.center = self.center + dt * self.velocity
        for ax in range(3):                                # reflect at the box walls
            if abs(self.center[ax] - self.home[ax]) > self.box_half:
                self.center[ax] = self.home[ax] + np.sign(self.center[ax]-self.home[ax])*self.box_half
                self.velocity[ax] *= -1

# =========================================================== 5. MPC ===========
class MPController:
    """Receding-horizon MPC (kinematic). Each tick it solves a small QP that is
    the SCP linearisation of: track the A->B reference, respect speed limits,
    and keep every arm control point clear of the (predicted) obstacle.
    Warm-started; ~10 ms/solve on a 6-DoF arm."""
    def __init__(self, checker, H=HORIZON, dt=DT, v_max=V_MAX, margin=MARGIN,
                 w_track=1.0, w_term=20.0, w_u=0.02):
        import scipy.sparse as sp
        self.sp = sp; import osqp; self.osqp = osqp
        self.checker, self.H, self.dt = checker, H, dt
        self.v_max, self.margin = v_max, margin
        # Δq_k = dt * sum_{j<k} u_j   ->   ΔQ = L @ U
        L = np.zeros((H*nv, H*nv))
        for k in range(1, H+1):
            for j in range(k): L[(k-1)*nv:k*nv, j*nv:(j+1)*nv] = np.eye(nv)
        self.L = L * dt
        Wd = np.concatenate([np.ones(nv)*w_track]*(H-1) + [np.ones(nv)*w_term])
        self.W = np.diag(Wd)
        P = 2*(self.L.T @ self.W @ self.L + w_u*np.eye(H*nv))
        self.P = sp.csc_matrix((P + P.T)/2)
        self.u_prev = None

    def _rollout(self, q0, U):
        qs, dq, acc = [], np.zeros(self.H*nv), np.zeros(nv)
        for k in range(self.H):
            acc = acc + self.dt * U[k*nv:(k+1)*nv]
            dq[k*nv:(k+1)*nv] = acc
            qs.append(pin.integrate(model, q0, acc))
        return qs, dq

    def solve(self, q0, ref_idx, obstacle):
        H, dt = self.H, self.dt
        Unom = self.u_prev if self.u_prev is not None else np.zeros(H*nv)
        qs_nom, dq_nom = self._rollout(q0, Unom)
        # tracking cost target: advance along the reference over the horizon
        B = np.zeros(H*nv)
        for k in range(1, H+1):
            B[(k-1)*nv:k*nv] = pin.difference(model, q_ref[min(ref_idx+k, N_REF-1)], q0)
        qlin = 2*dt*(self.L.T @ self.W @ B)
        rows, lo, up = [np.eye(H*nv)], [-self.v_max*np.ones(H*nv)], [self.v_max*np.ones(H*nv)]
        # collision (SCP linearisation about the rollout, obstacle predicted const-vel)
        G, h = [], []
        cvel = obstacle.velocity
        for k in range(1, H+1):
            ck = obstacle.center + k*dt*cvel
            pred = type("O", (), {"center": ck, "radius": obstacle.radius})
            qk = qs_nom[k-1]; dqk = dq_nom[(k-1)*nv:k*nv]; Lk = self.L[(k-1)*nv:k*nv, :]
            for d, Jd in self.checker.query(qk, pred):
                G.append(-(Jd @ Lk)); h.append(d - self.margin - Jd @ dqk)
        if G:
            rows.append(np.array(G)); lo.append(-np.inf*np.ones(len(h))); up.append(np.array(h))
        A = self.sp.csc_matrix(np.vstack(rows))
        prob = self.osqp.OSQP()
        prob.setup(self.P, qlin, A, np.concatenate(lo), np.concatenate(up),
                   verbose=False, warm_start=True, max_iter=4000)
        if self.u_prev is not None: prob.warm_start(x=self.u_prev)
        res = prob.solve()
        U = res.x if res.info.status_val in (1, 2) else np.zeros(H*nv)
        self.u_prev = np.concatenate([U[nv:], U[-nv:]])   # shift for next warm start
        return U[:nv]

# =========================================================== 6. RUN ===========
def main():
    checker = make_checker(CHECKER)
    print(f"collision module: {type(checker).__name__}")
    # obstacle wanders through the region the arm sweeps (centred on the path)
    mid = 0.5*(poseA.translation + poseB.translation)
    obstacle = DynamicObstacle(mid + np.array([0.22, 0.0, 0.10]),
                               box_half=0.26, speed=0.16)

    q = qA.copy(); mpc = MPController(checker)
    traj, obs_c, clear, solve_ms = [], [], [], []
    for t in range(SIM_STEPS):
        ref_idx = min(int(t / REF_SPAN * (N_REF-1)), N_REF-1)
        t0 = time.time(); u0 = mpc.solve(q, ref_idx, obstacle); solve_ms.append(1e3*(time.time()-t0))
        q = pin.integrate(model, q, DT*u0)
        dmin = min(d for d, _ in checker.query(q, obstacle))
        traj.append(q.copy()); obs_c.append(obstacle.center.copy()); clear.append(dmin)
        obstacle.step(DT)
    traj = np.array(traj); obs_c = np.array(obs_c); clear = np.array(clear)

    reached = np.linalg.norm(fk_pose(q).translation - poseB.translation)
    print(f"min clearance: {clear.min():+.3f} m  (collision if <0)   "
          f"penetration: {clear.min()<0}")
    print(f"reached B: EE error {reached:.3f} m   median solve: {np.median(solve_ms):.1f} ms")
    _render(traj, obs_c, clear, solve_ms)

def _render(traj, obs_c, clear, solve_ms):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    def arm_pts(q):
        p = arm_control_points(q); return np.array([[0,0,0]] + p)
    ee = np.array([fk_pose(q).translation for q in traj])
    fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2*np.pi, 14); vv = np.linspace(0, np.pi, 7)
    su, sv = np.outer(np.cos(u), np.sin(vv)), np.outer(np.sin(u), np.sin(vv)); sw = np.outer(np.ones_like(u), np.cos(vv))
    def draw(k):
        ax.cla()
        P = arm_pts(traj[k])
        ax.plot([poseA.translation[0],poseB.translation[0]],
                [poseA.translation[1],poseB.translation[1]],
                [poseA.translation[2],poseB.translation[2]], "--", color="#bbb", lw=1.3, label="A→B reference")
        ax.plot(ee[:k+1,0], ee[:k+1,1], ee[:k+1,2], color="#e07b39", lw=2, label="EE (MPC)")
        ax.plot(P[:,0], P[:,1], P[:,2], "-o", color="#2b6cb0", lw=4, ms=5)
        c = obs_c[k]; col = "#d33" if clear[k] < 0 else "#2a9d8f"
        ax.plot_surface(c[0]+OBST_R*su, c[1]+OBST_R*sv, c[2]+OBST_R*sw, color=col, alpha=0.4)
        ax.scatter(*poseA.translation, color="green", s=45); ax.text(*poseA.translation, " A", color="green")
        ax.scatter(*poseB.translation, color="red", s=45);   ax.text(*poseB.translation, " B", color="red")
        ax.set_xlim(-0.5,0.7); ax.set_ylim(-0.6,0.5); ax.set_zlim(-0.1,0.7)
        ax.set_title(f"UR3 MPC dodging  |  step {k+1}/{len(traj)}  |  clearance {clear[k]*100:+.1f} cm")
        ax.legend(loc="upper left", fontsize=8); ax.view_init(elev=22, azim=-60)
    FuncAnimation(fig, draw, frames=len(traj), interval=60).save(
        "mpc_avoidance.gif", writer=PillowWriter(fps=18)); plt.close(fig)
    print("saved:", os.path.join(os.getcwd(), "mpc_avoidance.gif"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.axhline(0, color="#d33", lw=1); a1.axhline(MARGIN, color="#888", ls="--", lw=1, label="margin")
    a1.plot(clear, color="#2a9d8f", lw=2); a1.set_ylabel("min clearance [m]"); a1.legend(fontsize=8); a1.grid(alpha=.3)
    a2.plot(solve_ms, color="#2b6cb0", lw=1.5); a2.set_ylabel("solve time [ms]"); a2.set_xlabel("tick"); a2.grid(alpha=.3)
    fig.suptitle("MPC collision-avoidance metrics"); fig.tight_layout(); fig.savefig("mpc_metrics.png", dpi=120); plt.close(fig)
    print("saved:", os.path.join(os.getcwd(), "mpc_metrics.png"))

if __name__ == "__main__":
    main()
