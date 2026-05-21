FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/LimChanju/SeRT-Training-Platform"
LABEL org.opencontainers.image.description="SeRT VR Training Platform — dependency environment"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# 의존성 먼저 복사 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY pyproject.toml .
COPY v2/ ./v2/

# Isaac Sim 전용 모듈(omni.*, pxr)은 런타임에 없으므로 stub 생성
RUN python - <<'EOF'
import os, sys

# omni / pxr stub — CI 스모크 테스트 전용
for mod in ["omni", "omni.isaac", "omni.isaac.kit", "omni.isaac.core",
            "omni.isaac.core.utils", "omni.isaac.core.utils.viewports",
            "omni.isaac.core.objects", "omni.isaac.franka",
            "omni.isaac.franka.tasks", "omni.isaac.franka.controllers",
            "pxr"]:
    parts = mod.split(".")
    parent = None
    for i, part in enumerate(parts):
        full = ".".join(parts[:i+1])
        if full not in sys.modules:
            import types
            m = types.ModuleType(full)
            sys.modules[full] = m
            if parent:
                setattr(parent, part, m)
        parent = sys.modules[full]

stub_path = "/app/isaac_stubs"
os.makedirs(stub_path, exist_ok=True)

stub_code = '''
import sys, types

_STUBS = [
    "omni", "omni.isaac", "omni.isaac.kit", "omni.isaac.core",
    "omni.isaac.core.utils", "omni.isaac.core.utils.viewports",
    "omni.isaac.core.objects", "omni.isaac.franka",
    "omni.isaac.franka.tasks", "omni.isaac.franka.controllers",
    "pxr",
]

for mod in _STUBS:
    parts = mod.split(".")
    parent = None
    for i, part in enumerate(parts):
        full = ".".join(parts[:i+1])
        if full not in sys.modules:
            m = types.ModuleType(full)
            sys.modules[full] = m
            if parent:
                setattr(parent, part, m)
        parent = sys.modules[full]

class _AnyClass:
    def __init__(self, *a, **kw): pass
    def __getattr__(self, n): return _AnyClass()
    def __call__(self, *a, **kw): return _AnyClass()

import omni.isaac.kit as _kit
_kit.SimulationApp = _AnyClass
import omni.isaac.core as _core
_core.World = _AnyClass
import omni.isaac.core.utils.viewports as _vp
_vp.set_camera_view = lambda *a, **kw: None
import omni.isaac.core.objects as _obj
_obj.DynamicCuboid = _AnyClass
_obj.FixedCuboid   = _AnyClass
_obj.VisualCuboid  = _AnyClass
import pxr as _pxr
_pxr.Gf = _AnyClass()
'''

with open(f"{stub_path}/isaac_stubs.pth", "w") as f:
    f.write(stub_path + "\n")

with open(f"{stub_path}/sitecustomize.py", "w") as f:
    f.write(stub_code)

print("Isaac Sim stubs installed.")
EOF

# sitecustomize를 site-packages에 배치
RUN cp /app/isaac_stubs/sitecustomize.py \
       $(python -c "import site; print(site.getsitepackages()[0])")/sitecustomize.py

ENV PYTHONPATH="/app"

CMD ["python", "-c", "from v2 import __version__; print('sert-vr-training', __version__, 'OK')"]
