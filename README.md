# mobile-robotic-arm-pick-and-place

**Coupled dynamics · Operational-space control · Hierarchical quadratic programming · MuJoCo**

A physics-based simulation of a differential-drive mobile robot carrying a six-degree-of-freedom manipulator, developed to investigate coordinated base–arm motion during pick-and-place operations. Instead of controlling the mobile platform and manipulator independently, the project uses a reduced whole-body dynamic model and allocates end-effector acceleration across both subsystems while enforcing actuator and motion constraints.


## 1. Robot model

The configuration is

\[
q=[x,\ y,\ \theta,\ q_1,\ldots,q_6]^T\in\mathbb{R}^9,
\]

where \((x,y,\theta)\) describes the planar base and \(q_1,\ldots,q_6\) are the manipulator joint angles. Ideal differential-drive rolling prevents instantaneous lateral motion, so the admissible velocity vector is

\[
\eta=[v,\ \omega,\ \dot q_1,\ldots,\dot q_6]^T\in\mathbb{R}^8.
\]

The velocity basis \(S(\theta)\) imposes the rolling constraint directly:

\[
\dot q=S(\theta)\eta,\qquad
\ddot q=S(\theta)\dot\eta+\dot S(\theta)\eta.
\]

The \(\dot S\eta\) term is important during turning: even with constant forward and yaw speeds, the base accelerates in world coordinates as its heading changes.

## 2. Coupled reduced dynamics

Starting with the full constrained dynamics,

\[
M(q)\ddot q+C(q,\dot q)\dot q+G(q)
=B(q)u+A(q)^T\lambda_c,
\]

projecting into the admissible velocity basis eliminates ideal lateral constraint reactions:

\[
\boxed{\bar M(q)\dot\eta+b(q,\eta)=\bar B(q)u}
\]

with

\[
\bar M=S^TMS,\qquad
b=S^T\left(C\dot q+G+M\dot S\eta\right),\qquad
\bar B=S^TB.
\]

The reduced inertia \(\bar M\in\mathbb R^{8\times8}\) includes base–arm coupling. Consequently, accelerating the arm can require wheel torque even if the desired base acceleration is zero.

The MuJoCo model constructs the reduced mass matrix and bias from the simulator's full dynamics and an admissible velocity basis that also maps wheel speeds. The implementation checks positive definiteness of the reduced mass matrix and rank of the actuator mapping.

## 3. Six-dimensional operational-space tracking

The controlled task is the end-effector's world-frame position and orientation. A consistent hybrid twist uses world-frame linear velocity and tool-frame angular velocity:

\[
V_E=\bar J(q)\eta,\qquad \bar J\in\mathbb R^{6\times8}.
\]

The controller constructs desired translational and rotational accelerations from the reference pose, twist, acceleration, and measured errors. It compensates for Jacobian variation:

\[
\dot V_E=\bar J\dot\eta+\dot{\bar J}\eta,
\qquad
b_t=a^*_{task}-\dot{\bar J}\eta.
\]

Orientation error is calculated from rotation matrices rather than subtracting Euler angles. The implementation estimates Jacobian variation by finite differences along admissible motion.

## 4. Dynamic allocation and singularity handling

When the task Jacobian has full row rank, the mass-weighted allocation is

\[
\bar J_M^\#=\bar M^{-1}\bar J^T
\left(\bar J\bar M^{-1}\bar J^T\right)^{-1},
\]

\[
\dot\eta_{nom}=\bar J_M^\# b_t+
\left(I-\bar J_M^\#\bar J\right)z.
\]

The primary term tracks the end effector; the secondary term expresses preferred internal motion. The implementation uses a normalized, damped version of this allocation, monitors a normalized singular-value indicator, and includes secondary damping and bounded joint-limit repulsion. Near rank loss, the damped inverse is an approximation rather than an exact nullspace projector; hard task priorities are handled separately by the constrained optimizer.

## 5. Hierarchical quadratic programming

Each control cycle solves for the eight reduced accelerations. The code's optimization order is:

1. **Hard constraints:** torque, joint-position barrier conditions, reduced-velocity and wheel-speed barriers, and acceleration limits.
2. **End-effector tracking:** minimize the normalized six-dimensional task-acceleration residual.
3. **Base motion:** prefer forward/yaw accelerations toward the current base waypoint.
4. **Arm posture:** apply the implemented joint-posture preference where compatible with higher-priority outputs.
5. **Actuator effort:** minimize a normalized effort-related objective in remaining directions.

Higher-priority achieved outputs are retained as equalities for subsequent optimization levels. If a lower-priority solve fails, the implementation retains the preceding solution; a failure at the primary level raises a controller fault. **The actual code orders base preference before posture**, and the retained outputs and numerical tolerances determine practical priority preservation.

After optimization, requested actuator torques follow from inverse dynamics:

\[
\boxed{u=\bar B^{-1}(\bar M\dot\eta+b).}
\]

The controller checks the inverse-dynamics residual and rejects commands outside configured torque bounds. OSQP is used when available, with a SciPy SLSQP fallback.

## 6. Pick-and-place sequence

The separate sequencer provides smooth pose references, base waypoints, gripper commands, phase transitions, and fault conditions. The supplied implementation uses these phases:

| Phase | Purpose |
|---|---|
| `READY` | Initialize at the current pose. |
| `RAISE` | Raise the tool to a clearance pose. |
| `APPROACH` | Move toward the pick station. |
| `ALIGN` | Refine the grasp pose. |
| `CLOSE` | Close the gripper and verify two-sided contact. |
| `LIFT` | Lift the object and verify clearance and settling. |
| `CLEAR_STATION` | Follow a predefined clearance waypoint away from the pick station. |
| `TRANSPORT` | Coordinate base and arm toward the destination. |
| `PRE_PLACE` | Approach the placement pose. |
| `PLACE` | Lower the object and verify table support. |
| `RELEASE` | Open the gripper and check placement. |
| `RETREAT` | Withdraw from the placed object. |

Translational trajectories use a quintic time-scaling function; orientation follows rotation-vector interpolation. The implementation uses **preplanned clearance waypoints, not a general-purpose collision planner**.

After grasp confirmation, an additional payload model contributes mass and inertia terms to the reduced dynamics. The model is detached when the release phase begins. Contact checks monitor gripper-pad forces, object support, and unexpected robot–table contact.

## 7. Further development

Potential extensions include collision-aware waypoint generation, explicit wheel–ground traction constraints, online payload identification, benchmarking under model uncertainty, . These are future directions, not claims about the current implementation.


