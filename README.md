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
