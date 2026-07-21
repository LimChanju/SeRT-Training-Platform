"""PhysX-backed hand-to-Panda distal-link safety queries.

The robot geometry is discovered from ``UsdPhysics.CollisionAPI`` on the
composed Stage.  No visual mesh, hand-authored capsule, or guessed link radius
is used.  A bounded binary search over PhysX sphere-overlap queries finds the
signed gap from the tracked hand sphere to the nearest distal collider.
"""

from __future__ import annotations

import json
import math
import os
import time
import numpy as np

try:
    from v3_chan.end_effector_safety_geometry import (
        DISTAL_LINK_NAMES,
        LINK_ID_BY_NAME,
        MISSING_SURFACE_GAP_M,
        EndEffectorSafetyResult,
        HandSafetyResult,
        SafetyThresholds,
        classify_surface_gap,
    )
except ImportError:
    from end_effector_safety_geometry import (
        DISTAL_LINK_NAMES,
        LINK_ID_BY_NAME,
        MISSING_SURFACE_GAP_M,
        EndEffectorSafetyResult,
        HandSafetyResult,
        SafetyThresholds,
        classify_surface_gap,
    )


class PandaEndEffectorSafetyRuntime:
    """Query the built-in Franka collision shapes against tracked hand spheres."""

    GEOMETRY_SOURCE = "isaac_stage_usdphysics_collisionapi_physx_overlap_sphere"

    def __init__(
        self,
        *,
        robot_prim_path: str = "/World/Franka",
        thresholds: SafetyThresholds | None = None,
        debug: bool | None = None,
    ) -> None:
        import omni.physx
        import omni.usd

        self.robot_prim_path = robot_prim_path.rstrip("/")
        self.thresholds = (thresholds or SafetyThresholds.from_env()).validated()
        self.debug = (
            _env_bool("DEBUG_HRI_SAFETY_GEOMETRY", False)
            if debug is None
            else bool(debug)
        )
        self.debug_visualization = _env_bool(
            "HRI_DEBUG_SAFETY_VISUALIZATION", False
        )
        self.debug_print_every = max(
            1, int(os.environ.get("HRI_SAFETY_DEBUG_PRINT_EVERY", "30"))
        )
        self._stage = omni.usd.get_context().get_stage()
        self._scene_query = omni.physx.get_physx_scene_query_interface()
        self._collider_to_link: dict[str, str] = {}
        self._collider_id_by_path: dict[str, int] = {}
        self._collider_properties: dict[str, dict[str, object]] = {}
        self._query_count = 0
        self._query_time_ms = 0.0
        self._query_error_logged = False
        self._debug_counts = {"left": 0, "right": 0}
        self._last_debug_state: dict[str, tuple | None] = {
            "left": None,
            "right": None,
        }
        self._debug_draw = None
        self._debug_bbox_cache = None
        self.refresh()
        if _env_bool("HRI_SHOW_PHYSX_COLLIDERS", False) or self.debug_visualization:
            self.enable_collider_visualization()
        if self.debug_visualization:
            self._init_debug_draw()

    @property
    def collider_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._collider_to_link))

    @property
    def available_links(self) -> tuple[str, ...]:
        found = set(self._collider_to_link.values())
        return tuple(name for name in DISTAL_LINK_NAMES if name in found)

    @property
    def missing_links(self) -> tuple[str, ...]:
        found = set(self._collider_to_link.values())
        return tuple(name for name in DISTAL_LINK_NAMES if name not in found)

    @property
    def geometry_valid(self) -> bool:
        return bool(self._collider_to_link)

    @property
    def mean_query_time_ms(self) -> float:
        return self._query_time_ms / max(1, self._query_count)

    def refresh(self) -> None:
        from pxr import PhysxSchema, Usd, UsdPhysics

        root = self._stage.GetPrimAtPath(self.robot_prim_path)
        if not root.IsValid():
            raise RuntimeError(f"Panda prim not found: {self.robot_prim_path}")

        collider_to_link: dict[str, str] = {}
        properties: dict[str, dict[str, object]] = {}
        for prim in Usd.PrimRange(root):
            collision_api = UsdPhysics.CollisionAPI(prim)
            if not collision_api:
                continue
            enabled = collision_api.GetCollisionEnabledAttr().Get()
            if enabled is False:
                continue
            path = str(prim.GetPath())
            link = _owning_distal_link(path)
            if link is None:
                continue
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            approximation = mesh_api.GetApproximationAttr().Get() if mesh_api else None
            physx_api = PhysxSchema.PhysxCollisionAPI(prim)
            contact_offset = (
                physx_api.GetContactOffsetAttr().Get() if physx_api else None
            )
            rest_offset = physx_api.GetRestOffsetAttr().Get() if physx_api else None
            collider_to_link[path] = link
            properties[path] = {
                "link": link,
                "prim_type": str(prim.GetTypeName()),
                "approximation": str(approximation) if approximation is not None else "",
                "contact_offset_m": _finite_or_none(contact_offset),
                "rest_offset_m": _finite_or_none(rest_offset),
            }

        self._collider_to_link = collider_to_link
        self._collider_id_by_path = {
            path: index + 1 for index, path in enumerate(sorted(collider_to_link))
        }
        self._collider_properties = properties
        if not self._collider_to_link:
            raise RuntimeError(
                f"No distal Panda CollisionAPI prims found below {self.robot_prim_path}"
            )
        print(
            "[SafetyGeometry] source=built-in-PhysX "
            f"colliders={len(self._collider_to_link)} "
            f"links={list(self.available_links)} missing={list(self.missing_links)} "
            f"hand_radius={self.thresholds.hand_radius_m:.3f}m "
            f"near_miss={self.thresholds.near_miss_gap_m:.3f}m "
            f"near={self.thresholds.near_gap_m:.3f}m "
            f"gate=[{self.thresholds.gate_full_gap_m:.3f},"
            f"{self.thresholds.gate_start_gap_m:.3f}]m",
            flush=True,
        )
        for path in sorted(self._collider_to_link):
            print(
                f"[SafetyGeometryCollider] id={self._collider_id_by_path[path]} "
                f"link={self._collider_to_link[path]} path={path} "
                f"props={properties[path]}",
                flush=True,
            )
        for link_name in self.missing_links:
            note = (
                " The built-in panda_hand collider covers the coincident flange "
                "frame; no guessed collider was added."
                if link_name == "panda_link8"
                else " No guessed collider was added."
            )
            print(
                f"[SafetyGeometry][WARNING] requested link={link_name} has no "
                f"enabled CollisionAPI prim below {self.robot_prim_path}.{note}",
                flush=True,
            )

    def evaluate(
        self,
        left_hand_pos,
        right_hand_pos,
    ) -> EndEffectorSafetyResult:
        result = EndEffectorSafetyResult(
            left=self.evaluate_hand("left", left_hand_pos),
            right=self.evaluate_hand("right", right_hand_pos),
        )
        if self.debug_visualization:
            self._draw_debug_result(result, left_hand_pos, right_hand_pos)
        return result

    def evaluate_hand(self, hand: str, hand_pos) -> HandSafetyResult:
        try:
            result = self._evaluate_hand(hand, hand_pos)
        except Exception as exc:
            if not self._query_error_logged:
                self._query_error_logged = True
                print(
                    f"[SafetyGeometry] PhysX query failed; marking geometry invalid: {exc}",
                    flush=True,
                )
            result = HandSafetyResult(hand=hand, geometry_valid=False)
        self._maybe_log_debug(result)
        return result

    def _evaluate_hand(self, hand: str, hand_pos) -> HandSafetyResult:
        pos = _valid_position(hand_pos)
        if pos is None or not self.geometry_valid:
            return HandSafetyResult(hand=hand, geometry_valid=False)

        started = time.perf_counter()
        query_count_before = self._query_count
        hand_radius = self.thresholds.hand_radius_m
        contact_hits = self._overlap_distal(hand_radius, pos)
        contact = bool(contact_hits)

        if contact:
            low = max(1e-5, self.thresholds.query_tolerance_m * 0.25)
            low_hits = self._overlap_distal(low, pos)
            if low_hits:
                nearest_radius = 0.0
                final_hits = low_hits
            else:
                high = hand_radius
                final_hits = contact_hits
                for _ in range(self.thresholds.query_iterations):
                    mid = (low + high) * 0.5
                    hits = self._overlap_distal(mid, pos)
                    if hits:
                        high = mid
                        final_hits = hits
                    else:
                        low = mid
                nearest_radius = high
                final_hits = self._overlap_distal(
                    high + self.thresholds.query_tolerance_m, pos
                ) or final_hits
            surface_gap = nearest_radius - hand_radius
        else:
            low = hand_radius
            high = hand_radius + self.thresholds.max_query_gap_m
            broad_hits = self._overlap_distal(high, pos)
            if not broad_hits:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                return self._result(
                    hand=hand,
                    geometry_valid=True,
                    surface_gap_m=MISSING_SURFACE_GAP_M,
                    hits=(),
                    contact=False,
                    query_time_ms=elapsed_ms,
                    query_count=self._query_count - query_count_before,
                )
            final_hits = broad_hits
            for _ in range(self.thresholds.query_iterations):
                mid = (low + high) * 0.5
                hits = self._overlap_distal(mid, pos)
                if hits:
                    high = mid
                    final_hits = hits
                else:
                    low = mid
            surface_gap = high - hand_radius
            final_hits = self._overlap_distal(
                high + self.thresholds.query_tolerance_m, pos
            ) or final_hits

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = self._result(
            hand=hand,
            geometry_valid=True,
            surface_gap_m=surface_gap,
            hits=final_hits,
            contact=contact,
            query_time_ms=elapsed_ms,
            query_count=self._query_count - query_count_before,
        )
        return result

    def metadata(self) -> dict[str, object]:
        thresholds = self.thresholds
        return {
            "safety_geometry_source": self.GEOMETRY_SOURCE,
            "safety_robot_prim_path": self.robot_prim_path,
            "safety_requested_links_json": json.dumps(DISTAL_LINK_NAMES),
            "safety_available_links_json": json.dumps(self.available_links),
            "safety_missing_links_json": json.dumps(self.missing_links),
            "safety_missing_link_notes_json": json.dumps(
                {
                    "panda_link8": (
                        "The Isaac Sim 4.5 Franka asset has no CollisionAPI below "
                        "panda_link8; its coincident physical flange is covered by "
                        "the built-in panda_hand collider."
                    )
                }
                if "panda_link8" in self.missing_links
                else {},
                sort_keys=True,
            ),
            "safety_link_id_map_json": json.dumps(LINK_ID_BY_NAME, sort_keys=True),
            "safety_human_hand_id_map_json": json.dumps(
                {"left": 1, "right": 2}, sort_keys=True
            ),
            "safety_collider_id_map_json": json.dumps(
                self._collider_id_by_path, sort_keys=True
            ),
            "safety_collider_properties_json": json.dumps(
                self._collider_properties, sort_keys=True
            ),
            "safety_hand_radius_m": thresholds.hand_radius_m,
            "safety_collision_gap_m": thresholds.collision_gap_m,
            "safety_near_miss_gap_m": thresholds.near_miss_gap_m,
            "safety_near_gap_m": thresholds.near_gap_m,
            "safety_gate_full_gap_m": thresholds.gate_full_gap_m,
            "safety_gate_start_gap_m": thresholds.gate_start_gap_m,
            "safety_max_query_gap_m": thresholds.max_query_gap_m,
            "safety_query_tolerance_m": thresholds.query_tolerance_m,
            "safety_query_iterations": thresholds.query_iterations,
            "safety_contact_signal": "physx_overlap_sphere_at_hand_radius",
            "safety_contact_force_available": False,
            "safety_contact_force_note": (
                "Tracked hands are non-physical query spheres; contact is exact "
                "PhysX overlap against Panda colliders and force is unavailable."
            ),
        }

    def _maybe_log_debug(self, result: HandSafetyResult) -> None:
        if not self.debug:
            return
        hand = result.hand
        self._debug_counts[hand] = self._debug_counts.get(hand, 0) + 1
        state = (
            result.geometry_valid,
            result.contact,
            result.collision,
            result.near_miss,
            result.near,
            result.closest_link,
        )
        previous = self._last_debug_state.get(hand)
        active = result.near or result.contact
        changed = previous is not None and state != previous
        first_relevant = previous is None and (active or not result.geometry_valid)
        periodic = active and self._debug_counts[hand] % self.debug_print_every == 0
        self._last_debug_state[hand] = state
        if not (changed or first_relevant or periodic):
            return
        print(
            "[SafetyGeometryDBG] "
            f"hand={hand} valid={int(result.geometry_valid)} "
            f"gap={result.surface_gap_m:.4f}m "
            f"contact={int(result.contact)} collision={int(result.collision)} "
            f"near_miss={int(result.near_miss)} "
            f"near={int(result.near)} gate={result.distance_gate:.3f} "
            f"link={result.closest_link or '-'} "
            f"collider={result.closest_collider_path or '-'} "
            f"queries={result.query_count} time={result.query_time_ms:.3f}ms",
            flush=True,
        )

    def _init_debug_draw(self) -> None:
        try:
            from omni.debugdraw import get_debug_draw_interface
            from pxr import Usd, UsdGeom

            self._debug_draw = get_debug_draw_interface()
            self._debug_bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            print(
                "[SafetyGeometry] closest-collider debug overlay enabled "
                "(exact PhysX colliders + hand status sphere + association line).",
                flush=True,
            )
        except Exception as exc:
            self._debug_draw = None
            self._debug_bbox_cache = None
            print(f"[SafetyGeometry] debug overlay unavailable: {exc}", flush=True)

    def _draw_debug_result(
        self,
        result: EndEffectorSafetyResult,
        left_hand_pos,
        right_hand_pos,
    ) -> None:
        if self._debug_draw is None:
            return
        from pxr import Gf

        for hand_result, position in (
            (result.left, left_hand_pos),
            (result.right, right_hand_pos),
        ):
            pos = _valid_position(position)
            if pos is None:
                continue
            color = _debug_color(hand_result)
            hand_point = Gf.Vec3f(*[float(value) for value in pos])
            try:
                self._debug_draw.draw_sphere(
                    hand_point,
                    float(self.thresholds.hand_radius_m * 1.02),
                    color,
                )
            except Exception:
                pass
            collider_center = self._collider_debug_center(
                hand_result.closest_collider_path
            )
            if collider_center is None:
                continue
            collider_point = Gf.Vec3f(
                *[float(value) for value in collider_center]
            )
            try:
                self._debug_draw.draw_line(
                    hand_point,
                    color,
                    collider_point,
                    color,
                )
                self._debug_draw.draw_point(collider_point, color, 12.0)
            except Exception:
                pass

    def _collider_debug_center(self, collider_path: str) -> np.ndarray | None:
        if not collider_path or self._debug_bbox_cache is None:
            return None
        prim = self._stage.GetPrimAtPath(collider_path)
        if not prim.IsValid():
            return None
        try:
            self._debug_bbox_cache.Clear()
            aligned_range = self._debug_bbox_cache.ComputeWorldBound(
                prim
            ).ComputeAlignedRange()
            center = (aligned_range.GetMin() + aligned_range.GetMax()) * 0.5
            return np.asarray(center, dtype=float)
        except Exception:
            return None

    def enable_collider_visualization(self) -> None:
        import carb
        from omni.physx import bindings

        settings = carb.settings.get_settings()
        settings.set(bindings._physx.SETTING_DISPLAY_COLLIDERS, True)
        settings.set(bindings._physx.SETTING_DISPLAY_COLLIDER_NORMALS, False)
        print(
            "[SafetyGeometry] PhysX collider visualization enabled "
            "(HRI_SHOW_PHYSX_COLLIDERS=1).",
            flush=True,
        )

    def _overlap_distal(self, radius: float, pos: np.ndarray) -> tuple[str, ...]:
        hits: set[str] = set()

        def _report(hit) -> bool:
            path = str(hit.collision)
            if path in self._collider_to_link:
                hits.add(path)
            return True

        started = time.perf_counter()
        self._scene_query.overlap_sphere(
            float(max(radius, 1e-6)),
            tuple(float(value) for value in pos),
            _report,
            False,
        )
        self._query_count += 1
        self._query_time_ms += (time.perf_counter() - started) * 1000.0
        return tuple(sorted(hits))

    def _result(
        self,
        *,
        hand: str,
        geometry_valid: bool,
        surface_gap_m: float,
        hits,
        contact: bool,
        query_time_ms: float,
        query_count: int,
    ) -> HandSafetyResult:
        hit_paths = tuple(sorted(path for path in hits if path in self._collider_to_link))
        closest_path = hit_paths[0] if hit_paths else ""
        closest_link = self._collider_to_link.get(closest_path, "")
        classification = classify_surface_gap(
            surface_gap_m,
            self.thresholds,
            contact=contact,
            geometry_valid=geometry_valid,
        )
        return HandSafetyResult(
            hand=hand,
            geometry_valid=geometry_valid,
            surface_gap_m=float(surface_gap_m),
            closest_link=closest_link,
            closest_link_id=LINK_ID_BY_NAME.get(closest_link, 0),
            closest_collider_path=closest_path,
            closest_collider_id=self._collider_id_by_path.get(closest_path, 0),
            contact=bool(contact),
            collision=classification.collision,
            contact_force_n=0.0,
            contact_force_valid=False,
            penetration_m=max(0.0, -float(surface_gap_m)),
            near_miss=classification.near_miss,
            near=classification.near,
            distance_gate=classification.distance_gate,
            query_time_ms=float(query_time_ms),
            query_count=int(query_count),
        )


def _owning_distal_link(path: str) -> str | None:
    parts = set(path.split("/"))
    return next((name for name in DISTAL_LINK_NAMES if name in parts), None)


def _valid_position(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def _finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _debug_color(result: HandSafetyResult) -> int:
    if not result.geometry_valid:
        return 0xFF808080
    if result.collision:
        return 0xFFFF3030
    if result.near_miss:
        return 0xFFFFB020
    if result.near:
        return 0xFFFFFF40
    if result.distance_gate > 0.0:
        return 0xFF40C8FF
    return 0xFF50D080


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
