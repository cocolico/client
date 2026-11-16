# GDK EEF execution adapter — 2026-10-15

## Current 61D client: combined head/arm commands

The original 17D adapter described below remains unchanged. The current
`run_ours_client.py --task yifu --model pi05 --execute` route uses
`execute_sharpa61` instead. No hardware execution was started for this update.

- The first chunk has a measured-state-to-first-target smoothstep approach of
  **at least 3 seconds**, extended for larger EEF/hand deltas. Then all 50 frames
  execute at 20 Hz with 100 Hz interpolation; later chunks keep the old timing.
- Each 61D subframe sends one official `TrajectoryTrackingControl` protobuf via
  GDK Publisher on `/gdk/model_predict`. Factory arbitration and native arm IK
  remain in the path. Arms use absolute `base_link` xyz+xyzw; head uses three
  scalar absolute joint groups yaw/roll/pitch (idx11/12/13; groups 0/9/1).
- The old separate head/arm sends are not used in this route. The active robot
  arbitrator log confirms `/gdk/joint_control` priority 40 and
  `/gdk/end_effector_pose_control` priority 41 share group `control`. This is a
  concrete source of head-command starvation. Combined model-predict uses the
  existing factory route at priority 82; no arbitrator settings were edited.
- The GDK 3.3.8 high-level dict adapter inserts default zero elbow reference
  poses. The new sender omits those optional references instead of adding extra
  IK constraints. It does not implement local IK or disable protection.
- Sharpa sending, observations, inference, local validation and cancellation
  checks are unchanged. Send errors propagate; there is no fallback to competing
  separate head/arm requests.

Offline tests cover real protobuf serialization, all five control groups,
head ordering, no elbow references, publisher reuse, failure-before-hand-send,
the smooth approach and full 50-frame rollout. They use fake hardware transports.

Hardware acceptance remains pending. A successful publish is not confirmation
of robot execution. The combined factory trajectory route schedules a one-frame
sample over 10 ms (speed 1); its internal lifetime appears to be 10 times that
interval, unlike the old explicit 50 ms EEF lifetime. GDK default QoS is retained
(Reliable, KeepLast depth 5, TransientLocal); reconnect behavior has not been
hardware-validated. Do not claim identical native timeout/reconnect behavior.
Factory VR takeover still has higher priority; do not concurrently run teleop.

Head-fix backup: `/home/descfly/code/agibot-head-combined-backup-L6pFmj/`.
Startup-transition backup: `/home/descfly/code/agibot-startup-backup-lf9WcD/`.

## Original 17D adapter

Scope: execution-layer replacement only. No robot was started. Sharpa, the
61-dimensional model/client routing, and camera changes are not part of this step.

`AgiBotRobot.execute_eef(actions, start_eef, start_joints, check=None)` accepts
1–32 rows of 17 values: left xyz + quaternion xyzw, right xyz + quaternion xyzw,
and three head joint positions. EEF poses are absolute in `base_link`, in metres;
head positions are radians. Both start_eef and named start_joints must come from
the observation used for inference, not model targets. The head interpolation
start is taken from named joint feedback.

Arms use `EndEffectorPoseControl` / `kBothArms`, exactly the command mechanism of
`scripts/replay_full_eef.py`; GDK owns arm IK. Head uses
`move_head_joint_servo(positions, 0.02, False)`. There are no hand or gripper sends.
Head and arm sends are separate, not atomic; hardware coexistence and head joint
ordering must be confirmed in a later authorized hardware test. This adapter is
not yet wired into the existing 19D inference client.

The 20 Hz -> 100 Hz schedule remains five ticks per target. Positions and head
angles interpolate linearly; orientation uses shortest-arc quaternion SLERP.
EEF lifetime is 0.05 s, matching the reference replay script. No local IK is used.

Preserved: finite/shape validation, 0.1 rad pre-execution observation drift check
on head/arms, 100 ms lateness stop, cancellation callback, existing head boundary
handling, and 0.35 rad head target-step check. Per user approval, no arm target
joint-step check is possible/applied in the EEF path. The original 19D joint path,
including its arm step check and gripper behavior, is unchanged. No GDK protection
settings were modified; no additional EEF tracking/residual threshold was added.

Superseded local IK and its tests/dependency note are archived outside the repo:
`/home/descfly/code/agibot-gdk-eef-backup-20261015-hqL2KC/`.
The previously installed SciPy package was not uninstalled.

Offline verification (does not initialize GDK):

```bash
cd /home/descfly/code/agibot_inference
PYTHONPATH=scripts:/home/descfly/.cache/agibot/app/gdk/lib \
  /home/descfly/miniconda3/envs/gdk-py310/bin/python \
  -m unittest discover -s tests -v
```
