"""Pure helpers for HRI experiment-level HDF5 metadata."""

from __future__ import annotations

from collections.abc import Mapping


_INVALID_TEXT = {"", "unknown", "unspecified", "none", "null"}
EXPERIMENT_METADATA_SCHEMA_VERSION = "hri_experiment_metadata_v3"
_VALID_HANDEDNESS = {"left", "right", "ambidextrous", "other"}


def _text(environment: Mapping[str, str], name: str, default: str) -> str:
    value = str(environment.get(name, default)).strip()
    return value or default


def _bool(environment: Mapping[str, str], name: str, default: bool) -> int:
    value = str(environment.get(name, str(int(default)))).strip().lower()
    return int(value in ("1", "true", "yes", "on"))


def _float(environment: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(environment.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _int(environment: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(environment.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def build_experiment_metadata(
    *,
    environment: Mapping[str, str],
    participant_id: str,
    participant_session_index: int,
    participant_handedness: str,
    is_practice: bool | int,
    experiment_condition: str,
    experiment_block_id: str,
    haptic_condition: str,
    protocol_version: str,
    room_calibration_id: str,
    xr_mode: str,
    xr_backend: str,
    haptics_intensity: int,
    haptics_min_interval_s: float,
    haptics_contact_min_steps: int,
    haptics_udp_port: int,
    haptics_enabled: bool | int,
    haptics_udp_configured: bool,
    collection_max_episodes: int,
    sample_interval_steps: int,
    task_success_xy_tolerance_m: float,
    task_success_z_tolerance_m: float,
    task_success_max_speed_mps: float,
    task_success_min_lift_m: float,
) -> dict[str, str | int | float]:
    """Return reproducibility metadata without importing Isaac Sim."""

    return {
        "experiment_metadata_schema_version": EXPERIMENT_METADATA_SCHEMA_VERSION,
        "participant_id": str(participant_id).strip() or "unspecified",
        "participant_id_kind": "pseudonym",
        "participant_session_index": int(participant_session_index),
        "participant_handedness": str(participant_handedness).strip().lower()
        or "unspecified",
        "is_practice": int(is_practice),
        "experiment_condition": str(experiment_condition).strip() or "unspecified",
        "experiment_block_id": str(experiment_block_id).strip() or "unspecified",
        "haptic_experiment_condition": str(haptic_condition).strip().lower()
        or "unspecified",
        "protocol_version": str(protocol_version).strip() or "unspecified",
        "collection_protocol_version": str(protocol_version).strip() or "unspecified",
        "room_calibration_id": str(room_calibration_id).strip() or "unspecified",
        "collection_test_mode": _bool(environment, "HRI_COLLECTION_TEST_MODE", False),
        "collection_max_episodes": int(collection_max_episodes),
        "sample_interval_steps": int(sample_interval_steps),
        "task_success_xy_tolerance_m": float(task_success_xy_tolerance_m),
        "task_success_z_tolerance_m": float(task_success_z_tolerance_m),
        "task_success_max_speed_mps": float(task_success_max_speed_mps),
        "task_success_min_lift_m": float(task_success_min_lift_m),
        "xr_mode": str(xr_mode).strip() or "unspecified",
        "xr_backend": str(xr_backend).strip() or "unspecified",
        "xr_controller_pose_mode": _text(
            environment, "XR_CONTROLLER_POSE_MODE", "unspecified"
        ),
        "xr_external_hand_tracking": _bool(
            environment, "XR_EXTERNAL_HAND_TRACKING", False
        ),
        "xr_hand_tracking_udp_port": _int(
            environment, "HAND_TRACKING_UDP_PORT", 5555
        ),
        "xr_hand_proxy_enabled": _bool(environment, "XR_HAND_PROXY_ENABLED", False),
        "xr_hand_sphere_enabled": _bool(environment, "XR_HAND_SPHERE_ENABLED", True),
        "xr_hand_haptic_point_mode": _text(
            environment, "XR_HAND_HAPTIC_POINT_MODE", "sphere"
        ),
        "xr_hand_proxy_radius_m": _float(
            environment, "HRI_HAND_PROXY_RADIUS_M", 0.035
        ),
        "xr_show_controllers": _bool(environment, "XR_SHOW_CONTROLLERS", True),
        "xr_controller_workspace_guard": _bool(
            environment, "XR_CONTROLLER_WORKSPACE_GUARD", True
        ),
        "xr_controller_max_head_dist_m": _float(
            environment, "XR_CONTROLLER_MAX_HEAD_DIST_M", 1.85
        ),
        "xr_controller_min_z_m": _float(environment, "XR_CONTROLLER_MIN_Z_M", 0.35),
        "xr_controller_max_z_m": _float(environment, "XR_CONTROLLER_MAX_Z_M", 2.25),
        "xr_hand_visual_offset": _text(environment, "XR_HAND_VISUAL_OFFSET", "0,0,0"),
        "xr_left_hand_visual_offset": _text(
            environment, "XR_LEFT_HAND_VISUAL_OFFSET", "0,0,0"
        ),
        "xr_right_hand_visual_offset": _text(
            environment, "XR_RIGHT_HAND_VISUAL_OFFSET", "0,0,0"
        ),
        "haptics_transport": "udp",
        "haptics_enabled": int(haptics_enabled),
        "haptics_udp_configured": int(bool(haptics_udp_configured)),
        "haptics_udp_port": int(haptics_udp_port),
        "haptics_event": "panda_distal_surface_contact",
        "haptics_intensity": int(haptics_intensity),
        "haptics_intensity_scale": "0_to_100",
        "haptics_min_interval_s": float(haptics_min_interval_s),
        "haptics_contact_min_steps": int(haptics_contact_min_steps),
        "haptics_pulse_flag_semantics": (
            "haptic_pulse_left/right=1 only when UDP sendto succeeds; "
            "bridge receipt and physical glove vibration are not confirmed"
        ),
        "haptics_device_confirmation_available": 0,
    }


def validate_experiment_metadata(
    *,
    production_mode: bool,
    participant_id: str,
    participant_session_index: int,
    participant_handedness: str,
    is_practice: bool | int,
    experiment_condition: str,
    experiment_block_id: str,
    haptic_condition: str,
    haptics_enabled: bool | int,
    haptics_udp_configured: bool | int,
    protocol_version: str,
    room_calibration_id: str,
) -> None:
    """Reject incomplete study metadata before a production session starts."""

    if not production_mode:
        return
    required = {
        "HRI_PARTICIPANT_ID": participant_id,
        "HRI_PARTICIPANT_HANDEDNESS": participant_handedness,
        "HRI_EXPERIMENT_CONDITION": experiment_condition,
        "HRI_EXPERIMENT_BLOCK_ID": experiment_block_id,
        "HRI_HAPTIC_CONDITION": haptic_condition,
        "HRI_PROTOCOL_VERSION": protocol_version,
        "HRI_ROOM_CALIBRATION_ID": room_calibration_id,
    }
    missing = [
        name
        for name, value in required.items()
        if str(value or "").strip().lower() in _INVALID_TEXT
    ]
    if missing:
        raise RuntimeError(
            "Production collection requires experiment metadata: "
            + ", ".join(missing)
        )
    if int(participant_session_index) < 1:
        raise RuntimeError(
            "Production collection requires HRI_PARTICIPANT_SESSION_INDEX >= 1"
        )
    normalized_handedness = str(participant_handedness).strip().lower()
    if normalized_handedness not in _VALID_HANDEDNESS:
        raise RuntimeError(
            "HRI_PARTICIPANT_HANDEDNESS must be one of "
            + ", ".join(sorted(_VALID_HANDEDNESS))
        )
    if int(is_practice) not in (0, 1):
        raise RuntimeError(
            "Production collection requires HRI_IS_PRACTICE to be explicitly 0 or 1"
        )

    normalized_haptic_condition = str(haptic_condition).strip().lower()
    if normalized_haptic_condition not in {"on", "off"}:
        raise RuntimeError(
            "Production collection requires HRI_HAPTIC_CONDITION to be 'on' or 'off'"
        )
    enabled = int(haptics_enabled)
    if enabled not in (0, 1):
        raise RuntimeError(
            "Production collection requires BHAPTICS_ENABLED to be explicitly 0 or 1"
        )
    expected_enabled = int(normalized_haptic_condition == "on")
    if enabled != expected_enabled:
        raise RuntimeError(
            "HRI_HAPTIC_CONDITION does not match BHAPTICS_ENABLED "
            f"({normalized_haptic_condition!r} versus {enabled})"
        )
    udp_configured = int(haptics_udp_configured)
    if udp_configured not in (0, 1):
        raise RuntimeError(
            "Production collection requires haptics_udp_configured to be 0 or 1"
        )
    if enabled != udp_configured:
        raise RuntimeError(
            "BHAPTICS_ENABLED does not match BHAPTICS_NOTEBOOK_IP configuration"
        )
