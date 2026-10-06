# UR3e Cartesian Velocity MPC

Run commands from this directory after activating the environment that contains
`numpy`, `scipy`, `osqp`, `pinocchio`, and `loguru`:

```bash
conda activate MPC_test
```

## Basic simulation

```bash
python ur3e_cartesian_velocity_mpc.py
```


## Meshcat and a static box obstacle

This additionally requires `meshcat`, `coal`, and `Pillow` (only when saving
a GIF):

```bash
python ur3e_cartesian_velocity_mpc.py --meshcat --no-plot \
  --obstacle-center 0.45 0.13 0.303 \
  --obstacle-size 0.06 0.06 0.06 \
  --tool-collision-radius 0.03 \
  --collision-margin 0.04
```

`--obstacle-center` is the absolute world-frame position of the box centre.
`--obstacle-size` supplies its full X/Y/Z side lengths in metres, so equal
values create a cube. `--tool-collision-radius` is the tool sphere radius;
`--collision-margin` is the additional required surface clearance. Start with
the obstacle outside the margin at the initial robot pose, otherwise the hard
QP can be infeasible.

## Detour planning

Add `--plan-detour` to test the direct route and then collision-check simple
two-waypoint routes around each box face. The shortest valid route becomes the
MPC reference; the MPC's link/tool clearance constraints stay active while it
tracks that reference. Orange markers in Meshcat are selected waypoints.

```bash
python ur3e_cartesian_velocity_mpc.py --meshcat --no-plot \
  --q0-deg 0 -90 90 -90 -90 0 \
  --delta-position 0.10 0 0 \
  --obstacle-center 0.34 0.19 0.303 \
  --obstacle-size 0.02 0.02 0.02 \
  --tool-collision-radius 0.02 \
  --collision-margin 0.02 \
  --detour-clearance 0 \
  --plan-detour
```

`--detour-clearance` is optional extra planning clearance beyond
`--collision-margin`; it does not replace the hard MPC clearance constraint.
This is a static-box, simulation-only local planner, not a general global
planner for arbitrary scenes.

## Configuration-space planning

For a more difficult blocked path, use `--plan-cspace`. It runs a
bidirectional RRT-Connect planner in the six joint angles. Every sampled joint
edge is checked against the static box using the same all-link/tool GJK
clearance model used by MPC. The resulting joint path is converted to a
Cartesian pose/twist reference for MPC to track.

```bash
python ur3e_cartesian_velocity_mpc.py --meshcat --no-plot \
  --q0-deg 0 -90 90 -90 -90 0 \
  --delta-position 0.10 0 0 \
  --obstacle-center 0.34 0.19 0.303 \
  --obstacle-size 0.02 0.02 0.02 \
  --tool-collision-radius 0.02 \
  --collision-margin 0.02 \
  --plan-cspace
```

The orange dotted line in Meshcat is the planned tool path; it is not a
safety boundary. `--cspace-max-iterations` and `--cspace-joint-step` tune the
planner if it cannot find a route. Start and goal configurations must already
meet the collision margin: no planner can safely end at a target inside the
obstacle's clearance region.

The green GJK witness spheres default to a 3 mm radius. For smaller markers,
for example, add `--meshcat-witness-radius 0.0015`.

To save a new GIF (the output folder must already exist):

```bash
python ur3e_cartesian_velocity_mpc.py --meshcat --no-plot \
  --save-meshcat-gif animations/run.gif \
  --obstacle-center 0.35 0.24 0.303 \
  --obstacle-size 0.06 0.06 0.06
```

## Hardware

First validate the same target in simulation.  Hardware mode requires an
explicit confirmation and `ur_rtde`:

```bash
python ur3e_cartesian_velocity_mpc.py --hardware --confirm-hardware
```

Static-obstacle collision MPC and Meshcat are intentionally simulation-only.
