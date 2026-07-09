import os
import time

from isaacsim import SimulationApp


ISAACSIM_ROOT = os.environ.get("ISAACSIM_ROOT", os.path.expanduser("~/isaac-sim-4.5.0"))
EXPERIENCE = os.path.join(ISAACSIM_ROOT, "apps", "isaacsim.exp.base.xr.openxr.kit")
FRAMES = int(os.environ.get("OPENXR_HAND_CHECK_FRAMES", "900"))
PRINT_EVERY = max(1, int(os.environ.get("OPENXR_HAND_CHECK_PRINT_EVERY", "60")))

simulation_app = SimulationApp({"headless": False}, experience=EXPERIENCE)

from isaacsim.xr.openxr import OpenXR, OpenXRSpec


def _count_valid_joints(joints):
    if not joints:
        return 0, 0
    valid_pos = 0
    valid_pose = 0
    for joint in joints:
        if joint is None:
            continue
        flags = joint.locationFlags
        if flags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT:
            valid_pos += 1
        if (
            flags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT
            and flags & OpenXRSpec.XR_SPACE_LOCATION_ORIENTATION_VALID_BIT
        ):
            valid_pose += 1
    return valid_pos, valid_pose


def _sample_joint(joints):
    if not joints:
        return "None"
    for idx, joint in enumerate(joints):
        if joint is None:
            continue
        flags = joint.locationFlags
        if not (flags & OpenXRSpec.XR_SPACE_LOCATION_POSITION_VALID_BIT):
            continue
        pos = joint.pose.position
        return f"idx={idx} pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}) flags={int(flags)}"
    return "None"


openxr = OpenXR()
left_hand = OpenXRSpec.XrHandEXT.XR_HAND_LEFT_EXT
right_hand = OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT

print("[OpenXRHandCheck] Starting.")
print(f"[OpenXRHandCheck] ISAACSIM_ROOT={ISAACSIM_ROOT}")
print(f"[OpenXRHandCheck] EXPERIENCE={EXPERIENCE}")
print(f"[OpenXRHandCheck] XR_RUNTIME_JSON={os.environ.get('XR_RUNTIME_JSON', '<unset>')}")
print("[OpenXRHandCheck] Put Quest 3 in hand-tracking mode, remove controllers, and move hands in view.")

first_valid_logged = False
for frame in range(FRAMES):
    simulation_app.update()

    left_joints = openxr.locate_hand_joints(left_hand, stage_axis=True)
    right_joints = openxr.locate_hand_joints(right_hand, stage_axis=True)
    left_pos_count, left_pose_count = _count_valid_joints(left_joints)
    right_pos_count, right_pose_count = _count_valid_joints(right_joints)

    if frame % PRINT_EVERY == 0 or (
        not first_valid_logged and (left_pos_count > 0 or right_pos_count > 0)
    ):
        print(
            "[OpenXRHandCheck] "
            f"frame={frame} "
            f"left={left_pos_count}/26 pos {left_pose_count}/26 pose "
            f"right={right_pos_count}/26 pos {right_pose_count}/26 pose "
            f"left_sample={_sample_joint(left_joints)} "
            f"right_sample={_sample_joint(right_joints)}"
        )
    if left_pos_count > 0 or right_pos_count > 0:
        first_valid_logged = True
    time.sleep(0.005)

if first_valid_logged:
    print("[OpenXRHandCheck] RESULT: hand joints are reaching Isaac Sim OpenXR.")
else:
    print("[OpenXRHandCheck] RESULT: no valid OpenXR hand joints observed.")

simulation_app.close()
