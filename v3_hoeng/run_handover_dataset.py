from __future__ import annotations

import argparse
import os
import sys


ISAAC_ROOT = "/home/railab/isaac-sim-4.5.0"
DEFAULT_DATASET_DIR = "data/trajectories/handover_dataset_v0"


HANDOVER_DATASET_PRESET = {
    "ISAACSIM_ROOT": ISAAC_ROOT,
    "ENABLE_HANDOVER_TASK": "1",
    "ENABLE_PICK_PLACE": "0",
    "ENABLE_HANDOVER_DATASET_RECORDING": "1",
    "HANDOVER_DATASET_DIR": DEFAULT_DATASET_DIR,
    "HANDOVER_DATASET_OVERWRITE": "0",
    "HANDOVER_DATASET_RECORD_FAILED": "1",
    "HANDOVER_DATASET_MIN_STEPS": "20",
    "HANDOVER_DATASET_RECORD_IMAGES": "0",
    "HANDOVER_DATASET_SAVE_ON_INTERRUPT": "1",
    "HANDOVER_DATASET_AUTO_COMPLETE_ON_PLACE": "1",
    "HANDOVER_DATASET_AUTO_COMPLETE_STABLE_STEPS": "8",
    "HANDOVER_DATASET_PLACE_SUCCESS_DIST_M": "0.08",
    "XR_PHYSICAL_HAND_PROXY": "1",
    "XR_PHYSICAL_HAND_FILTER_ROBOT": "0",
    "XR_CONTROLLER_POSE_MODE": "visual",
    "XR_CONTROLLER_WORKSPACE_GUARD": "1",
    "XR_CONTROLLER_MAX_HEAD_DIST_M": "1.85",
    "XR_VIRTUAL_WORLD_POSE_FALLBACK": "1",
    "XR_STAGE_VISUAL_FALLBACK": "1",
    "XR_PHYSICAL_HAND_CONTACT_GRAB": "1",
    "XR_PHYSICAL_HAND_FIXED_JOINT_GRAB": "1",
    "XR_PHYSICAL_HAND_JOINT_COLLISION": "0",
    "XR_PHYSICAL_HAND_ALLOW_LOGICAL_FALLBACK": "0",
    "XR_PHYSICAL_HAND_CONTACT_MARGIN": "0.025",
    "XR_PHYSICAL_HAND_PROXY_RADIUS": "0.025",
    "HANDOVER_RESPAWN_TO_HAND": "0",
    "HANDOVER_ADOPT_GRABBED_CUBE": "1",
    "HANDOVER_HUMAN_CUBE_PHYSICS": "1",
    "HANDOVER_PHYSICAL_MODE": "1",
    "HANDOVER_FOLLOW_HAND_UNTIL_GRABBED": "0",
    "HANDOVER_STABLE_WINDOW_STEPS": "12",
    "HANDOVER_STABLE_MOTION_THRESHOLD_M": "0.035",
    "HANDOVER_STABLE_MAX_STEP_M": "0.055",
    "HANDOVER_STABLE_MEAN_STEP_M": "0.010",
    "HANDOVER_STABLE_JITTER_THRESHOLD_M": "0.025",
    "HANDOVER_STABLE_OUTLIER_TOLERANCE_M": "0.055",
    "HANDOVER_STABLE_REQUIRED_WINDOWS": "3",
    "STICKY_PROXIMITY_GRAB": "1",
    "STICKY_PROXIMITY_RADIUS": "0.08",
    "STICKY_PROXIMITY_INITIAL_GRAB": "0",
    "HANDOVER_TAKEOVER_ARM_BY_DISTANCE": "0",
    "HANDOVER_ROBOT_APPROACH_COLLISION": "1",
    "HANDOVER_ROBOT_GRASP_CONFIRM_EVENT": "4",
    "HANDOVER_ROBOT_FIXED_JOINT_GRASP": "0",
    "HANDOVER_ROBOT_GRASP_MAX_DIST_M": "0.085",
    "HANDOVER_ROBOT_GRASP_CLOSED_WIDTH_M": "0.075",
    "HANDOVER_ROBOT_PLACE_DONE_EVENT": "8",
    "HANDOVER_ROBOT_DISABLE_CUBE_COLLISION_DURING_CARRY": "0",
    "HANDOVER_ORIENTED_GRASP": "1",
    "HANDOVER_APPROACH_OFFSET_M": "0.008",
    "HANDOVER_GRASP_ATTACH_DIST_M": "0.08",
    "HANDOVER_GRASP_ATTACH_CENTER_DIST_M": "0.065",
    "HANDOVER_GRASP_ATTACH_WIDTH_M": "0.06",
    "HANDOVER_GRASP_JAW_AXIS_MAX_M": "0.04",
    "HANDOVER_GRASP_VERTICAL_MAX_M": "0.07",
    "HANDOVER_GRASP_HAND_TO_FINGERTIP_M": "0.103",
    "HANDOVER_GRASP_FINGERTIP_OFFSET_M": "0.045",
    "HANDOVER_GRASP_DEPTH_AXIS_MAX_M": "0.04",
    "HANDOVER_GRASP_HEAD_CLEARANCE_M": "0.075",
    "HANDOVER_GRASP_VERIFY_STEPS": "18",
    "HANDOVER_GRASP_VERIFY_TIMEOUT_STEPS": "90",
    "HANDOVER_GRASP_VERIFY_MAX_DROP_M": "0.04",
    "HANDOVER_GRASP_RETRY_ENABLED": "1",
    "HANDOVER_GRASP_MAX_RETRIES": "6",
    "HANDOVER_GRASP_RETRY_OPEN_STEPS": "24",
    "HANDOVER_GRASP_HUMAN_RELEASE_TIMEOUT_STEPS": "12",
    "HANDOVER_GRASP_APPROACH_TIMEOUT_STEPS": "180",
    "HANDOVER_MOVE_REACHED_TOLERANCE_M": "0.035",
    "HANDOVER_PHYSICAL_LIFT_M": "0.14",
    "HANDOVER_PHYSICAL_PLACE_ABOVE_M": "0.04",
    "HANDOVER_PHYSICAL_RELEASE_HOLD_STEPS": "30",
    "HANDOVER_PHYSICAL_CARRY_LOST_STEPS": "18",
    "HANDOVER_PHYSICAL_CARRY_LOST_CENTER_M": "0.12",
    "HANDOVER_PHYSICAL_STAGE_TIMEOUT_STEPS": "180",
    "HANDOVER_DISABLE_HUMAN_HAND_COLLISION_ON_ROBOT_GRASP": "1",
    "HANDOVER_ROBOT_FINGERTIP_JOINT_TAKEOVER": "1",
    "HANDOVER_HIGH_FRICTION_GRASP": "1",
    "HANDOVER_GRASP_STATIC_FRICTION": "2.5",
    "HANDOVER_GRASP_DYNAMIC_FRICTION": "2.0",
    "HANDOVER_SIDE_GRASP_WHEN_HAND_BELOW": "0",
    "GRAB_POSE_PRIORITY": "pinch,grip,aim,palm,poke",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the tuned VR handover dataset collection preset."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Saved dataset episodes before exit. Use 0 to keep running.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing dataset directory before recording.",
    )
    parser.add_argument(
        "--no-record-failed",
        action="store_true",
        help="Do not save failed handover episodes.",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=20,
        help="Minimum recorded steps required before saving an episode.",
    )
    parser.add_argument(
        "--record-images",
        action="store_true",
        help="Request image path recording in the dataset when synchronized images are available.",
    )
    parser.add_argument(
        "--place-success-dist",
        type=float,
        default=0.08,
        help="Cube-to-target distance threshold for dataset success validation.",
    )
    parser.add_argument(
        "--auto-complete-stable-steps",
        type=int,
        default=8,
        help="Stable target-placement steps required before auto-completing a dataset episode.",
    )
    parser.add_argument(
        "--release-hold-steps",
        type=int,
        default=30,
        help="Steps to hold the open gripper before place completion.",
    )
    parser.add_argument(
        "--sim-max-steps",
        type=int,
        default=0,
        help="Optional global simulation step cap. 0 disables the cap.",
    )
    args = parser.parse_args()

    for key, value in HANDOVER_DATASET_PRESET.items():
        os.environ.setdefault(key, value)
    os.environ["HANDOVER_DATASET_EXIT_AFTER_EPISODES"] = str(max(0, args.episodes))
    os.environ["HANDOVER_DATASET_DIR"] = str(args.dataset_dir)
    os.environ["HANDOVER_DATASET_OVERWRITE"] = "1" if args.overwrite else "0"
    os.environ["HANDOVER_DATASET_RECORD_FAILED"] = "0" if args.no_record_failed else "1"
    os.environ["HANDOVER_DATASET_MIN_STEPS"] = str(max(1, args.min_steps))
    os.environ["HANDOVER_DATASET_RECORD_IMAGES"] = "1" if args.record_images else "0"
    os.environ["HANDOVER_DATASET_PLACE_SUCCESS_DIST_M"] = str(max(0.005, args.place_success_dist))
    os.environ["HANDOVER_DATASET_AUTO_COMPLETE_STABLE_STEPS"] = str(
        max(1, args.auto_complete_stable_steps)
    )
    os.environ["HANDOVER_PHYSICAL_RELEASE_HOLD_STEPS"] = str(max(1, args.release_hold_steps))
    if args.sim_max_steps > 0:
        os.environ["SIM_MAX_STEPS"] = str(args.sim_max_steps)

    # Keep Isaac Kit from seeing this wrapper's CLI flags.
    sys.argv = [sys.argv[0]]

    import main as handover_main

    handover_main.main()


if __name__ == "__main__":
    main()
