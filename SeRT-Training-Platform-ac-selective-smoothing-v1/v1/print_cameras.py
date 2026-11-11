import os
from pxr import Usd, UsdGeom

usd_path = os.path.expanduser("~/isaac_vr_project/stack_blocks_with_human.usd")
print(f"Opening {usd_path}...", flush=True)

stage = Usd.Stage.Open(usd_path)

print("🚨 [디버그] 카메라 및 큐브 검색 중...", flush=True)

for prim in stage.Traverse():
    path = prim.GetPath().pathString
    if prim.IsA(UsdGeom.Camera):
        xform = UsdGeom.Xformable(prim)
        translate = xform.GetLocalTransformation().ExtractTranslation()
        print(f"📷 발견된 카메라: {path}")
        print(f"   현재 위치: {translate}")
    
    if "Green" in path or "Cube" in path:
        print(f"🟩 발견된 큐브/Green: {path}")
