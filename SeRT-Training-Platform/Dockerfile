# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Isaac Sim stub 생성 (CI/클라우드 환경용)
RUN python - <<'EOF'
import os

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

import omni.isaac.kit as _kit; _kit.SimulationApp = _AnyClass
import omni.isaac.core as _core; _core.World = _AnyClass
import omni.isaac.core.utils.viewports as _vp; _vp.set_camera_view = lambda *a, **kw: None
import omni.isaac.core.objects as _obj
_obj.DynamicCuboid = _AnyClass; _obj.FixedCuboid = _AnyClass; _obj.VisualCuboid = _AnyClass
import pxr as _pxr; _pxr.Gf = _AnyClass()
'''

os.makedirs("/install/isaac_stubs", exist_ok=True)
with open("/install/isaac_stubs/sitecustomize.py", "w") as f:
    f.write(stub_code)
with open("/install/isaac_stubs/isaac_stubs.pth", "w") as f:
    f.write("/install/isaac_stubs\n")
print("stubs OK")
EOF

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/LimChanju/SeRT-Training-Platform"
LABEL org.opencontainers.image.description="SeRT VR Training Platform — dependency environment"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# builder에서 설치된 패키지만 복사 (소스 제외)
COPY --from=builder /install /usr/local

# 소스 복사
COPY pyproject.toml .
COPY v2/ ./v2/
COPY api/ ./api/ 2>/dev/null || true

# stub을 site-packages에 연결
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    cp /usr/local/isaac_stubs/sitecustomize.py "$SITE/sitecustomize.py" && \
    echo "/usr/local/isaac_stubs" >> "$SITE/isaac_stubs.pth"

ENV PYTHONPATH="/app"
ENV PORT=8080

EXPOSE 8080

CMD ["python", "-m", "api.app"]
