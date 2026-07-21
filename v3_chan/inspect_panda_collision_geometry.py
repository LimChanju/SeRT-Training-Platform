"""Dump the collision geometry shipped with Isaac Sim's Franka asset.

Run this through ``launch_isaac.sh``.  The script intentionally creates the
same Panda as the pick-and-place collector, then discovers colliders from the
composed USD Stage instead of relying on guessed prim paths.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("ISAAC_XR_MODE", "off")
os.environ.setdefault("ISAAC_HEADLESS", "1")

from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp({"headless": True})

import omni.physx
import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from panda_robot import PANDA_PRIM_PATH, add_panda
from scene_setup import create_world
from end_effector_safety_geometry import DISTAL_LINK_NAMES
from end_effector_safety_runtime import PandaEndEffectorSafetyRuntime


def _owning_link(path: str) -> str | None:
    parts = path.split("/")
    for name in DISTAL_LINK_NAMES:
        if name in parts:
            return name
    return None


def _attr_value(api, getter: str):
    try:
        value = getattr(api, getter)().Get()
    except Exception:
        return None
    return value


def main() -> None:
    world = create_world()
    add_panda(world)
    world.reset()
    world.play()
    for _ in range(5):
        world.step(render=False)

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(PANDA_PRIM_PATH)
    if not root.IsValid():
        raise RuntimeError(f"Panda prim not found: {PANDA_PRIM_PATH}")

    colliders = []
    print("[PandaColliderInspect] composed Stage colliders", flush=True)
    for prim in Usd.PrimRange(root):
        collision_api = UsdPhysics.CollisionAPI(prim)
        if not collision_api:
            continue
        path = str(prim.GetPath())
        enabled = _attr_value(collision_api, "GetCollisionEnabledAttr")
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        approximation = (
            _attr_value(mesh_api, "GetApproximationAttr") if mesh_api else None
        )
        physx_api = PhysxSchema.PhysxCollisionAPI(prim)
        contact_offset = (
            _attr_value(physx_api, "GetContactOffsetAttr") if physx_api else None
        )
        rest_offset = (
            _attr_value(physx_api, "GetRestOffsetAttr") if physx_api else None
        )
        imageable = UsdGeom.Imageable(prim)
        purpose = imageable.GetPurposeAttr().Get() if imageable else None
        link = _owning_link(path)
        colliders.append((path, link))
        print(
            "[PandaCollider] "
            f"link={link or '-'} path={path} type={prim.GetTypeName()} "
            f"enabled={enabled} purpose={purpose} approximation={approximation} "
            f"contact_offset={contact_offset} rest_offset={rest_offset}",
            flush=True,
        )

    print(
        f"[PandaColliderInspect] total={len(colliders)} "
        f"distal={sum(link is not None for _, link in colliders)}",
        flush=True,
    )
    for link_name in DISTAL_LINK_NAMES:
        paths = [path for path, link in colliders if link == link_name]
        print(
            f"[PandaColliderLink] {link_name}: "
            f"{len(paths)} collider(s) {paths}",
            flush=True,
        )

    link8_prim = stage.GetPrimAtPath(f"{PANDA_PRIM_PATH}/panda_link8")
    hand_prim = stage.GetPrimAtPath(f"{PANDA_PRIM_PATH}/panda_hand")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link8_translation = (
        xform_cache.GetLocalToWorldTransform(link8_prim).ExtractTranslation()
        if link8_prim.IsValid()
        else None
    )
    hand_translation = (
        xform_cache.GetLocalToWorldTransform(hand_prim).ExtractTranslation()
        if hand_prim.IsValid()
        else None
    )
    print(
        f"[PandaColliderLink] panda_link8 prim_valid={link8_prim.IsValid()} "
        f"type={link8_prim.GetTypeName() if link8_prim.IsValid() else ''} "
        f"children={[str(child.GetPath()) for child in link8_prim.GetChildren()] if link8_prim.IsValid() else []} "
        f"world_t={link8_translation} hand_world_t={hand_translation}",
        flush=True,
    )

    scene_query = omni.physx.get_physx_scene_query_interface()
    query_methods = sorted(
        name
        for name in dir(scene_query)
        if any(token in name for token in ("overlap", "sweep", "raycast"))
    )
    print(f"[PandaColliderInspect] scene-query methods={query_methods}", flush=True)

    runtime = PandaEndEffectorSafetyRuntime(robot_prim_path=PANDA_PRIM_PATH)
    expected_available = set(DISTAL_LINK_NAMES) - {"panda_link8"}
    if set(runtime.available_links) != expected_available:
        raise AssertionError(
            f"Unexpected distal collider coverage: {runtime.available_links}"
        )
    if runtime.missing_links != ("panda_link8",):
        raise AssertionError(f"Unexpected missing links: {runtime.missing_links}")
    if link8_translation is None or hand_translation is None:
        raise AssertionError("panda_link8 or panda_hand transform is unavailable")
    if not np.allclose(
        np.asarray(link8_translation, dtype=float),
        np.asarray(hand_translation, dtype=float),
        atol=1e-6,
    ):
        raise AssertionError("panda_link8 and panda_hand transforms do not coincide")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    contact_query_times = []
    for path in runtime.collider_paths:
        prim = stage.GetPrimAtPath(path)
        world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        center = np.asarray(
            (world_range.GetMin() + world_range.GetMax()) * 0.5,
            dtype=float,
        )
        result = runtime.evaluate_hand("smoke", center)
        if not result.contact or result.closest_collider_path != path:
            raise AssertionError(
                f"Collider smoke failed: target={path} result={result}"
            )
        contact_query_times.append(result.query_time_ms)
        print(
            "[PandaColliderSmoke] "
            f"target={path} center={np.round(center, 4)} "
            f"contact={result.contact} gap={result.surface_gap_m:.5f} "
            f"closest_link={result.closest_link} "
            f"closest_collider={result.closest_collider_path} "
            f"queries={result.query_count} time_ms={result.query_time_ms:.3f}",
            flush=True,
        )
    print(
        "[PandaColliderContactBenchmark] "
        f"samples={len(contact_query_times)} "
        f"mean_ms={float(np.mean(contact_query_times)):.4f} "
        f"max_ms={float(np.max(contact_query_times)):.4f}",
        flush=True,
    )

    invalid = runtime.evaluate(None, np.array([np.nan, 0.0, 0.0]))
    if invalid.geometry_valid or invalid.collision or invalid.near:
        raise AssertionError(f"Tracking-loss smoke failed: {invalid}")
    print(
        "[PandaColliderTrackingLoss] geometry_valid=0 collision=0 near=0",
        flush=True,
    )

    benchmark_pos = np.array([1.1, 0.0, 0.8], dtype=float)
    benchmark_count = 120
    started = time.perf_counter()
    for _ in range(benchmark_count):
        runtime.evaluate(benchmark_pos, benchmark_pos)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        "[PandaColliderBenchmark] "
        f"hand_pos={benchmark_pos.tolist()} frames={benchmark_count} total_ms={elapsed_ms:.3f} "
        f"per_frame_ms={elapsed_ms / benchmark_count:.3f} "
        f"mean_physx_query_ms={runtime.mean_query_time_ms:.4f}",
        flush=True,
    )

    physics_frames = 240
    started = time.perf_counter()
    for _ in range(physics_frames):
        world.step(render=False)
    baseline_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(physics_frames):
        world.step(render=False)
        runtime.evaluate(benchmark_pos, benchmark_pos)
    queried_seconds = time.perf_counter() - started
    print(
        "[PandaColliderFPS] "
        f"frames={physics_frames} "
        f"baseline_fps={physics_frames / baseline_seconds:.1f} "
        f"with_queries_fps={physics_frames / queried_seconds:.1f} "
        f"baseline_ms={baseline_seconds * 1000.0 / physics_frames:.3f} "
        f"with_queries_ms={queried_seconds * 1000.0 / physics_frames:.3f}",
        flush=True,
    )

    world.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
