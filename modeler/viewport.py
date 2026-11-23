"""OpenGL 3.3 Core Profile 3D Viewport with studio lighting, picking, and 3D transform gizmo."""

from __future__ import annotations

import ctypes
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from .math3d import (
    closest_point_on_axis,
    distance_point_to_segment_2d,
    look_at,
    normalize,
    orthographic,
    perspective,
    project_point,
    ray_aabb_intersect,
    ray_plane_intersect,
    ray_triangle_intersect,
    rotate_euler_xyz,
    screen_to_ray,
    transform_points,
)
from .mesh import Edge
from .scene import EditMode, Scene, SceneObject, ShadingMode, TransformTool


# ---------------------------------------------------------------------------
# GLSL 330 Core Shaders
# ---------------------------------------------------------------------------

BG_VERT_SRC = """
#version 330 core
const vec2 POS[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);
out vec2 vUv;
void main() {
    vec2 p = POS[gl_VertexID];
    vUv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.9999, 1.0);
}
"""

BG_FRAG_SRC = """
#version 330 core
in vec2 vUv;
out vec4 fragColor;
void main() {
    vec3 bottom = vec3(0.115, 0.115, 0.118);
    vec3 top    = vec3(0.195, 0.195, 0.200);
    float t = clamp(vUv.y, 0.0, 1.0);
    // Subtle radial vignette
    vec2 d = vUv - vec2(0.5);
    float vig = 1.0 - dot(d, d) * 0.24;
    vec3 col = mix(bottom, top, smoothstep(0.0, 1.0, t)) * vig;
    fragColor = vec4(col, 1.0);
}
"""

MESH_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;

out vec3 vWorldPos;
out vec3 vWorldNormal;

void main() {
    vec4 wp = uModel * vec4(aPos, 1.0);
    vWorldPos = wp.xyz;
    vWorldNormal = normalize(uNormalMat * aNormal);
    gl_Position = uProj * uView * wp;
}
"""

MESH_FRAG_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;

uniform vec3 uCameraPos;
uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform float uSelectedMix;
uniform vec3 uSelectedColor;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vWorldNormal);
    if (!gl_FrontFacing) {
        N = -N;
    }
    vec3 V = normalize(uCameraPos - vWorldPos);

    // Three-point studio lighting
    vec3 L1 = normalize(vec3(0.55, 0.85, 0.65));   // Key (warm)
    vec3 L2 = normalize(vec3(-0.75, 0.35, 0.45));  // Fill (cool)
    vec3 L3 = normalize(vec3(-0.20, 0.55, -0.85)); // Rim

    float ndl1 = max(dot(N, L1), 0.0);
    float ndl2 = max(dot(N, L2), 0.0);
    float ndl3 = max(dot(N, L3), 0.0);

    // Hemisphere ambient
    float hemi = N.y * 0.5 + 0.5;
    vec3 ambient = mix(vec3(0.10, 0.12, 0.16), vec3(0.24, 0.28, 0.36), hemi);

    vec3 diffuse =
        ndl1 * vec3(1.00, 0.96, 0.90) * 0.92 +
        ndl2 * vec3(0.62, 0.76, 0.98) * 0.36 +
        ndl3 * vec3(0.70, 0.80, 0.95) * 0.24;

    // Specular highlight (Blinn-Phong scaled by roughness/metallic)
    float shininess = mix(112.0, 8.0, clamp(uRoughness, 0.05, 1.0));
    vec3 H1 = normalize(L1 + V);
    vec3 H2 = normalize(L2 + V);
    float spec1 = pow(max(dot(N, H1), 0.0), shininess);
    float spec2 = pow(max(dot(N, H2), 0.0), shininess * 0.7);
    vec3 specTint = mix(vec3(1.0), uBaseColor * 1.15, uMetallic);
    float specStrength = mix(0.24, 0.95, uMetallic) * (1.0 - uRoughness * 0.65);
    vec3 specular = specTint * (spec1 * 1.05 + spec2 * 0.35) * specStrength;

    // Subtle Fresnel rim
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.5);
    vec3 rim = mix(vec3(0.32, 0.44, 0.64), uBaseColor, 0.35) * fresnel * 0.24;

    vec3 surfaceColor = uBaseColor * (ambient + diffuse * (1.0 - uMetallic * 0.28)) + specular + rim;
    surfaceColor = mix(surfaceColor, uSelectedColor, uSelectedMix);

    // ACES filmic tone mapping for rich, crisp studio contrast
    vec3 x = max(surfaceColor * 0.92, vec3(0.0));
    surfaceColor = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    fragColor = vec4(clamp(surfaceColor, 0.0, 1.0), 1.0);
}
"""

FLAT_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 uMVP;
uniform float uPointSize;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
}
"""

FLAT_FRAG_SRC = """
#version 330 core
uniform vec4 uColor;
uniform int uCirclePoint;
out vec4 fragColor;

void main() {
    if (uCirclePoint == 1) {
        vec2 c = gl_PointCoord - vec2(0.5);
        if (dot(c, c) > 0.25) {
            discard;
        }
    }
    fragColor = uColor;
}
"""


# ---------------------------------------------------------------------------
# Orbit Camera
# ---------------------------------------------------------------------------

class OrbitCamera:
    """Turntable orbit camera supporting perspective and orthographic projections."""

    def __init__(self) -> None:
        self.target = np.array([0.0, 0.6, 0.0], dtype=np.float32)
        self.distance: float = 6.2
        self.yaw: float = 38.0
        self.pitch: float = 26.0
        self.fov: float = 45.0
        self.orthographic: bool = False

    def reset(self) -> None:
        self.target = np.array([0.0, 0.6, 0.0], dtype=np.float32)
        self.distance = 6.2
        self.yaw = 38.0
        self.pitch = 26.0

    def eye(self) -> np.ndarray:
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)
        x = self.distance * math.cos(pitch_r) * math.sin(yaw_r)
        y = self.distance * math.sin(pitch_r)
        z = self.distance * math.cos(pitch_r) * math.cos(yaw_r)
        return (self.target + np.array([x, y, z], dtype=np.float32)).astype(np.float32)

    def forward(self) -> np.ndarray:
        return normalize(self.target - self.eye())

    def right(self) -> np.ndarray:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = normalize(np.cross(self.forward(), up))
        if np.linalg.norm(r) < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return r

    def up(self) -> np.ndarray:
        return normalize(np.cross(self.right(), self.forward()))

    def view_matrix(self) -> np.ndarray:
        return look_at(self.eye(), self.target, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def projection_matrix(self, aspect: float) -> np.ndarray:
        aspect = max(aspect, 1e-4)
        if self.orthographic:
            half_h = self.distance * math.tan(math.radians(self.fov) * 0.5)
            half_w = half_h * aspect
            return orthographic(-half_w, half_w, -half_h, half_h, -200.0, 400.0)
        return perspective(self.fov, aspect, 0.05, 400.0)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.45
        self.pitch = float(np.clip(self.pitch + dy * 0.45, -89.0, 89.0))

    def pan(self, dx: float, dy: float, height: int) -> None:
        h = max(float(height), 1.0)
        world_per_px = (2.0 * self.distance * math.tan(math.radians(self.fov) * 0.5)) / h
        delta = -self.right() * (dx * world_per_px) + self.up() * (dy * world_per_px)
        self.target = (self.target + delta).astype(np.float32)

    def zoom(self, wheel_steps: float) -> None:
        factor = math.pow(0.88, wheel_steps)
        self.distance = float(np.clip(self.distance * factor, 0.35, 180.0))

    def set_preset(self, preset: str) -> None:
        presets = {
            "front": (0.0, 0.0),
            "back": (180.0, 0.0),
            "right": (90.0, 0.0),
            "left": (-90.0, 0.0),
            "top": (0.0, 89.0),
            "bottom": (0.0, -89.0),
            "iso": (38.0, 26.0),
        }
        if preset in presets:
            self.yaw, self.pitch = presets[preset]

    def frame_bounds(self, bmin: np.ndarray, bmax: np.ndarray) -> None:
        center = (bmin + bmax) * 0.5
        radius = float(np.linalg.norm(bmax - bmin) * 0.5)
        self.target = center.astype(np.float32)
        self.distance = float(np.clip(max(radius * 2.6, 2.0), 0.8, 120.0))


# ---------------------------------------------------------------------------
# Shader Compilation Helper
# ---------------------------------------------------------------------------

def _compile_program(vert_src: str, frag_src: str) -> int:
    def compile_shader(src: str, shader_type: int) -> int:
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            raise RuntimeError(f"Shader compile error: {log}")
        return shader

    vs = compile_shader(vert_src, GL.GL_VERTEX_SHADER)
    fs = compile_shader(frag_src, GL.GL_FRAGMENT_SHADER)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
        raise RuntimeError(f"Shader link error: {log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog


# ---------------------------------------------------------------------------
# Viewport HUD Overlay (Transparent child widget — zero GL state conflicts)
# ---------------------------------------------------------------------------

class ViewportHUD(QWidget):
    """Crisp 2D HUD overlay displaying stats, mode badge, and orientation triad."""

    def __init__(self, viewport: "Viewport3D") -> None:
        super().__init__(viewport)
        self._vp = viewport
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w, h = self.width(), self.height()
        scene = self._vp.scene

        # Top-left Info Badge
        mode_labels = {
            EditMode.OBJECT: ("OBJECT MODE", QColor(212, 212, 212)),
            EditMode.VERTEX: ("EDIT: VERTEX", QColor(240, 160, 70)),
            EditMode.EDGE: ("EDIT: EDGE", QColor(240, 160, 70)),
            EditMode.FACE: ("EDIT: FACE", QColor(240, 160, 70)),
        }
        mode_text, mode_col = mode_labels[scene.edit_mode]
        proj_text = "Ortho" if self._vp.camera.orthographic else "Persp"
        tool_text = scene.tool.value.capitalize()

        total_v = sum(o.mesh.vertex_count for o in scene.objects if o.visible)
        total_f = sum(o.mesh.face_count for o in scene.objects if o.visible)
        active = scene.active_object()
        sel_line = f"Active: {active.name}" if active else "No Selection"
        if scene.edit_mode == EditMode.VERTEX:
            sel_line += f"  |  Verts: {len(scene.selected_vertices)}"
        elif scene.edit_mode == EditMode.EDGE:
            sel_line += f"  |  Edges: {len(scene.selected_edges)}"
        elif scene.edit_mode == EditMode.FACE:
            sel_line += f"  |  Faces: {len(scene.selected_faces)}"

        badge_rect = QRectF(12, 12, 265, 68)
        painter.setPen(QPen(QColor(255, 255, 255, 26), 1.0))
        painter.setBrush(QColor(24, 24, 24, 195))
        painter.drawRoundedRect(badge_rect, 8, 8)

        # Colored mode pill dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(mode_col)
        painter.drawEllipse(QPointF(24, 29), 4.5, 4.5)

        title_font = QFont("Sans Serif", 9, QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor(235, 235, 235))
        painter.drawText(34, 33, f"{mode_text}   ·   {tool_text}   ·   {proj_text}")

        sub_font = QFont("Sans Serif", 8)
        painter.setFont(sub_font)
        painter.setPen(QColor(175, 175, 175))
        painter.drawText(22, 52, sel_line)
        painter.drawText(
            22,
            68,
            f"Objects: {len(scene.objects)}   |   Verts: {total_v:,}   |   Faces: {total_f:,}",
        )

        # Top-right Orientation Triad
        cx, cy = w - 58.0, 58.0
        radius = 34.0
        painter.setPen(QPen(QColor(255, 255, 255, 24), 1.0))
        painter.setBrush(QColor(24, 24, 24, 175))
        painter.drawEllipse(QPointF(cx, cy), radius + 12, radius + 12)

        rot = self._vp.camera.view_matrix()[:3, :3]
        axes = [
            ("X", np.array([1.0, 0.0, 0.0], dtype=np.float32), QColor(235, 87, 87)),
            ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float32), QColor(88, 206, 114)),
            ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float32), QColor(82, 152, 245)),
        ]
        projected = []
        for label, vec, color in axes:
            cam_v = rot @ vec
            projected.append((float(cam_v[2]), label, float(cam_v[0]), float(-cam_v[1]), color))
        projected.sort(key=lambda item: item[0])

        axis_font = QFont("Sans Serif", 8, QFont.Weight.Bold)
        painter.setFont(axis_font)
        for _depth, label, px, py, color in projected:
            ex = cx + px * radius
            ey = cy + py * radius
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(ex, ey), 8.5, 8.5)
            painter.setPen(QColor(15, 18, 24))
            painter.drawText(
                QRectF(ex - 9, ey - 9, 18, 18),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

        # Bottom contextual hint strip
        if scene.edit_mode == EditMode.OBJECT:
            hint = "LMB: Select/Gizmo  |  MMB or Alt+LMB: Orbit  |  RMB or Shift+MMB: Pan  |  Wheel: Zoom  |  Tab: Edit Mode  |  G/R/S: Transform  |  F: Frame"
        else:
            hint = "LMB: Pick Element  |  Shift+LMB: Multi-Select  |  E: Extrude  |  I: Inset  |  1/2/3: Vert/Edge/Face  |  A: Select All  |  Tab: Object Mode"

        hint_font = QFont("Sans Serif", 8)
        painter.setFont(hint_font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(hint) + 24
        hx = max(12.0, (w - tw) * 0.5)
        hy = h - 32.0
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1.0))
        painter.setBrush(QColor(22, 22, 22, 185))
        painter.drawRoundedRect(QRectF(hx, hy, min(tw, w - 24), 22), 6, 6)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(
            QRectF(hx, hy, min(tw, w - 24), 22),
            Qt.AlignmentFlag.AlignCenter,
            hint,
        )
        painter.end()


# ---------------------------------------------------------------------------
# 3D Viewport Widget
# ---------------------------------------------------------------------------

class Viewport3D(QOpenGLWidget):
    """Interactive OpenGL 3.3 Core 3D viewport for modeling."""

    selection_changed = Signal()
    scene_modified = Signal()
    status_message = Signal(str)

    def __init__(self, scene: Scene, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.camera = OrbitCamera()

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # HUD overlay
        self._hud = ViewportHUD(self)

        # GL resources (initialized in initializeGL)
        self._gl_ready = False
        self._max_line_width: float = 1.0
        self._bg_prog = 0
        self._bg_vao = 0
        self._mesh_prog = 0
        self._flat_prog = 0

        # Dynamic reusable VAOs/VBOs
        self._mesh_vao = 0
        self._mesh_pos_vbo = 0
        self._mesh_nrm_vbo = 0
        self._line_vao = 0
        self._line_vbo = 0

        # Grid geometry buffers
        self._grid_minor_vao = 0
        self._grid_minor_count = 0
        self._grid_major_vao = 0
        self._grid_major_count = 0
        self._axes_vao = 0
        self._axes_vbo = 0

        # Mesh geometry cache: (obj_id, version, smooth) -> (pos, nrm, face_ids, edges)
        self._mesh_cache: Dict[
            Tuple[str, int, bool],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        # Interaction state
        self._drag_mode: Optional[str] = None  # "orbit", "pan", "gizmo"
        self._last_mouse = QPointF()
        self._press_mouse = QPointF()
        self._mouse_moved = False

        # Gizmo state
        self._hover_axis: Optional[str] = None  # "x", "y", "z", "free"
        self._active_axis: Optional[str] = None
        self._gizmo_start_pivot = np.zeros(3, dtype=np.float32)
        self._gizmo_start_param: float = 0.0
        self._gizmo_start_plane_hit = np.zeros(3, dtype=np.float32)
        self._gizmo_start_angle: float = 0.0
        self._gizmo_start_obj_transforms: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
        self._gizmo_start_edit_verts: Optional[np.ndarray] = None
        self._gizmo_edit_indices: List[int] = []

    # ------------------------------------------------------------------
    # Qt Lifecycle
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._hud.setGeometry(0, 0, self.width(), self.height())

    def refresh(self) -> None:
        self.update()
        self._hud.update()

    # ------------------------------------------------------------------
    # OpenGL Initialization & Resource Setup
    # ------------------------------------------------------------------

    def _set_line_width(self, width: float) -> None:
        # Strict OpenGL 3.3 Core Profile forbids glLineWidth > 1.0 on many drivers (Mesa, macOS).
        # Gizmos use solid 3D triangle geometry so line width can safely remain 1.0.
        clamped = max(1.0, min(float(width), self._max_line_width))
        if clamped > 1.0:
            try:
                GL.glLineWidth(clamped)
                return
            except Exception:
                self._max_line_width = 1.0
        GL.glLineWidth(1.0)

    def initializeGL(self) -> None:  # noqa: N802
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        self._max_line_width = 1.0

        self._bg_prog = _compile_program(BG_VERT_SRC, BG_FRAG_SRC)
        self._bg_vao = GL.glGenVertexArrays(1)

        self._mesh_prog = _compile_program(MESH_VERT_SRC, MESH_FRAG_SRC)
        self._flat_prog = _compile_program(FLAT_VERT_SRC, FLAT_FRAG_SRC)

        # Dynamic mesh VAO/VBOs
        self._mesh_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._mesh_vao)
        self._mesh_pos_vbo, self._mesh_nrm_vbo = GL.glGenBuffers(2)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_pos_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_nrm_vbo)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)

        # Dynamic line/point VAO/VBO
        self._line_vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._line_vao)
        self._line_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self._build_grid_buffers()
        self._gl_ready = True

    def _build_grid_buffers(self) -> None:
        extent = 16
        minor_pts: List[List[float]] = []
        major_pts: List[List[float]] = []

        for i in range(-extent, extent + 1):
            if i == 0:
                continue
            target = major_pts if (i % 4 == 0) else minor_pts
            target.append([float(i), 0.0, float(-extent)])
            target.append([float(i), 0.0, float(extent)])
            target.append([float(-extent), 0.0, float(i)])
            target.append([float(extent), 0.0, float(i)])

        self._grid_minor_vao, self._grid_minor_count = self._create_static_line_vao(
            np.asarray(minor_pts, dtype=np.float32)
        )
        self._grid_major_vao, self._grid_major_count = self._create_static_line_vao(
            np.asarray(major_pts, dtype=np.float32)
        )

        # World X and Z axes on ground plane
        axes_pts = np.array(
            [
                [-extent, 0.001, 0.0], [extent, 0.001, 0.0],  # X axis (0..2)
                [0.0, 0.001, -extent], [0.0, 0.001, extent],  # Z axis (2..4)
            ],
            dtype=np.float32,
        )
        self._axes_vao, _ = self._create_static_line_vao(axes_pts)

    def _create_static_line_vao(self, pts: np.ndarray) -> Tuple[int, int]:
        vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pts.nbytes, pts, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)
        return int(vao), int(len(pts))

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        dpr = self.devicePixelRatioF()
        GL.glViewport(0, 0, max(1, int(w * dpr)), max(1, int(h * dpr)))

    # ------------------------------------------------------------------
    # Rendering Helpers
    # ------------------------------------------------------------------

    def _get_cached_mesh(
        self, obj: SceneObject, smooth: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (obj.obj_id, obj.mesh.version, smooth)
        cached = self._mesh_cache.get(key)
        if cached is None:
            # Prune stale cache entries for this object
            for old_key in [k for k in self._mesh_cache if k[0] == obj.obj_id and k[1] != obj.mesh.version]:
                del self._mesh_cache[old_key]
            pos, nrm, face_ids = obj.mesh.triangulated_arrays(smooth=smooth)
            edges = obj.mesh.edge_positions()
            cached = (pos, nrm, face_ids, edges)
            self._mesh_cache[key] = cached
        return cached

    def _draw_lines(
        self,
        points: np.ndarray,
        mvp: np.ndarray,
        color: Tuple[float, float, float, float],
        mode: int = GL.GL_LINES,
        point_size: float = 1.0,
        circle_point: bool = False,
    ) -> None:
        if len(points) == 0:
            return
        pts = np.ascontiguousarray(points, dtype=np.float32)
        GL.glUseProgram(self._flat_prog)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._flat_prog, "uMVP"),
            1,
            GL.GL_TRUE,
            mvp,
        )
        GL.glUniform4f(GL.glGetUniformLocation(self._flat_prog, "uColor"), *color)
        GL.glUniform1f(GL.glGetUniformLocation(self._flat_prog, "uPointSize"), point_size)
        GL.glUniform1i(
            GL.glGetUniformLocation(self._flat_prog, "uCirclePoint"),
            1 if circle_point else 0,
        )

        GL.glBindVertexArray(self._line_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pts.nbytes, pts, GL.GL_DYNAMIC_DRAW)
        GL.glDrawArrays(mode, 0, len(pts))
        GL.glBindVertexArray(0)

    def paintGL(self) -> None:  # noqa: N802
        if not self._gl_ready:
            return

        dpr = self.devicePixelRatioF()
        GL.glViewport(0, 0, max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # 1. Studio gradient background
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self._bg_prog)
        GL.glBindVertexArray(self._bg_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)

        aspect = max(float(self.width()), 1.0) / max(float(self.height()), 1.0)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)
        vp = (proj @ view).astype(np.float32)
        cam_pos = self.camera.eye()

        # 2. Ground Grid & World Axes
        if self.scene.show_grid:
            GL.glUseProgram(self._flat_prog)
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._flat_prog, "uMVP"), 1, GL.GL_TRUE, vp
            )
            GL.glUniform1f(GL.glGetUniformLocation(self._flat_prog, "uPointSize"), 1.0)
            GL.glUniform1i(GL.glGetUniformLocation(self._flat_prog, "uCirclePoint"), 0)

            self._set_line_width(1.0)
            GL.glUniform4f(GL.glGetUniformLocation(self._flat_prog, "uColor"), 0.36, 0.36, 0.36, 0.28)
            GL.glBindVertexArray(self._grid_minor_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_minor_count)

            GL.glUniform4f(GL.glGetUniformLocation(self._flat_prog, "uColor"), 0.48, 0.48, 0.48, 0.44)
            GL.glBindVertexArray(self._grid_major_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_major_count)

            self._set_line_width(1.8)
            GL.glBindVertexArray(self._axes_vao)
            GL.glUniform4f(GL.glGetUniformLocation(self._flat_prog, "uColor"), 0.88, 0.30, 0.32, 0.85)
            GL.glDrawArrays(GL.GL_LINES, 0, 2)
            GL.glUniform4f(GL.glGetUniformLocation(self._flat_prog, "uColor"), 0.30, 0.56, 0.92, 0.85)
            GL.glDrawArrays(GL.GL_LINES, 2, 2)
            GL.glBindVertexArray(0)
            self._set_line_width(1.0)

        # 3. Scene Objects
        smooth = self.scene.shading == ShadingMode.SMOOTH
        selected_ids = set(self.scene.selected_object_ids)
        active_obj = self.scene.active_object()

        for obj in self.scene.objects:
            if not obj.visible:
                continue
            model = obj.model_matrix()
            mvp = (vp @ model).astype(np.float32)
            normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)
            pos, nrm, face_ids, edges = self._get_cached_mesh(obj, smooth=smooth)
            is_selected = obj.obj_id in selected_ids
            is_active = active_obj is not None and obj.obj_id == active_obj.obj_id

            # Solid Pass
            if self.scene.shading != ShadingMode.WIREFRAME and len(pos) > 0:
                GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
                GL.glPolygonOffset(1.0, 1.0)

                GL.glUseProgram(self._mesh_prog)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._mesh_prog, "uModel"), 1, GL.GL_TRUE, model)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._mesh_prog, "uView"), 1, GL.GL_TRUE, view)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._mesh_prog, "uProj"), 1, GL.GL_TRUE, proj)
                GL.glUniformMatrix3fv(GL.glGetUniformLocation(self._mesh_prog, "uNormalMat"), 1, GL.GL_TRUE, normal_mat)
                GL.glUniform3f(GL.glGetUniformLocation(self._mesh_prog, "uCameraPos"), float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
                GL.glUniform3f(GL.glGetUniformLocation(self._mesh_prog, "uBaseColor"), *obj.material.color)
                GL.glUniform1f(GL.glGetUniformLocation(self._mesh_prog, "uRoughness"), float(obj.material.roughness))
                GL.glUniform1f(GL.glGetUniformLocation(self._mesh_prog, "uMetallic"), float(obj.material.metallic))

                sel_mix = 0.14 if (is_selected and self.scene.edit_mode == EditMode.OBJECT) else 0.0
                GL.glUniform1f(GL.glGetUniformLocation(self._mesh_prog, "uSelectedMix"), sel_mix)
                GL.glUniform3f(GL.glGetUniformLocation(self._mesh_prog, "uSelectedColor"), 1.0, 0.62, 0.20)

                GL.glBindVertexArray(self._mesh_vao)
                pos_c = np.ascontiguousarray(pos, dtype=np.float32)
                nrm_c = np.ascontiguousarray(nrm, dtype=np.float32)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_pos_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, pos_c.nbytes, pos_c, GL.GL_DYNAMIC_DRAW)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_nrm_vbo)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, nrm_c.nbytes, nrm_c, GL.GL_DYNAMIC_DRAW)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(pos_c))
                GL.glBindVertexArray(0)
                GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

            # Wireframe Pass
            in_edit_here = self.scene.edit_mode != EditMode.OBJECT and is_active
            draw_wire = (
                self.scene.shading in (ShadingMode.WIREFRAME, ShadingMode.SOLID_WIRE)
                or (is_selected and self.scene.edit_mode == EditMode.OBJECT)
                or in_edit_here
            )
            if draw_wire and len(edges) > 0:
                if in_edit_here:
                    wire_col = (0.14, 0.16, 0.22, 0.90)
                    self._set_line_width(1.2)
                elif is_selected and self.scene.edit_mode == EditMode.OBJECT:
                    wire_col = (1.0, 0.62, 0.18, 0.95) if is_active else (0.95, 0.48, 0.15, 0.80)
                    self._set_line_width(1.6)
                elif self.scene.shading == ShadingMode.WIREFRAME:
                    wire_col = (*obj.material.color, 0.92)
                    self._set_line_width(1.3)
                else:
                    wire_col = (*obj.material.wireframe_color, 0.55)
                    self._set_line_width(1.0)
                self._draw_lines(edges, mvp, wire_col, GL.GL_LINES)
                self._set_line_width(1.0)

            # Edit Mode Sub-object Overlays for active object
            if in_edit_here:
                self._draw_edit_overlays(obj, mvp, pos, face_ids)

        # 4. Interactive 3D Transform Gizmo (always on top)
        pivot = self.scene.selection_pivot_world()
        if pivot is not None and self.scene.tool != TransformTool.SELECT:
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            self._draw_gizmo(pivot, vp)

    def _draw_edit_overlays(
        self,
        obj: SceneObject,
        mvp: np.ndarray,
        tri_pos: np.ndarray,
        face_ids: np.ndarray,
    ) -> None:
        mesh = obj.mesh
        mode = self.scene.edit_mode

        # Selected faces highlight
        if mode == EditMode.FACE and self.scene.selected_faces and len(tri_pos) > 0:
            mask = np.isin(face_ids, list(self.scene.selected_faces))
            sel_tris = tri_pos[mask]
            if len(sel_tris) > 0:
                GL.glDepthMask(GL.GL_FALSE)
                self._draw_lines(sel_tris, mvp, (1.0, 0.56, 0.14, 0.42), GL.GL_TRIANGLES)
                GL.glDepthMask(GL.GL_TRUE)

        # Selected edges highlight
        if mode == EditMode.EDGE and self.scene.selected_edges:
            edge_pts: List[np.ndarray] = []
            for u, v in self.scene.selected_edges:
                if 0 <= u < mesh.vertex_count and 0 <= v < mesh.vertex_count:
                    edge_pts.append(mesh.vertices[u])
                    edge_pts.append(mesh.vertices[v])
            if edge_pts:
                self._set_line_width(3.0)
                self._draw_lines(
                    np.asarray(edge_pts, dtype=np.float32),
                    mvp,
                    (1.0, 0.64, 0.16, 1.0),
                    GL.GL_LINES,
                )
                self._set_line_width(1.0)

        # Vertices display
        if mode == EditMode.VERTEX and mesh.vertex_count > 0:
            sel_mask = np.zeros(mesh.vertex_count, dtype=bool)
            for vi in self.scene.selected_vertices:
                if 0 <= vi < mesh.vertex_count:
                    sel_mask[vi] = True
            unsel_verts = mesh.vertices[~sel_mask]
            sel_verts = mesh.vertices[sel_mask]
            if len(unsel_verts) > 0:
                self._draw_lines(
                    unsel_verts,
                    mvp,
                    (0.85, 0.90, 0.98, 0.90),
                    GL.GL_POINTS,
                    point_size=7.0,
                    circle_point=True,
                )
            if len(sel_verts) > 0:
                self._draw_lines(
                    sel_verts,
                    mvp,
                    (1.0, 0.62, 0.14, 1.0),
                    GL.GL_POINTS,
                    point_size=10.5,
                    circle_point=True,
                )

        # Face center dots in Face mode for crisp visual feedback
        if mode == EditMode.FACE and mesh.face_count > 0:
            centers = np.array([mesh.face_center(i) for i in range(mesh.face_count)], dtype=np.float32)
            sel_mask = np.zeros(mesh.face_count, dtype=bool)
            for fi in self.scene.selected_faces:
                if 0 <= fi < mesh.face_count:
                    sel_mask[fi] = True
            if (~sel_mask).any():
                self._draw_lines(
                    centers[~sel_mask],
                    mvp,
                    (0.75, 0.82, 0.95, 0.72),
                    GL.GL_POINTS,
                    point_size=5.0,
                    circle_point=True,
                )
            if sel_mask.any():
                self._draw_lines(
                    centers[sel_mask],
                    mvp,
                    (1.0, 0.75, 0.20, 1.0),
                    GL.GL_POINTS,
                    point_size=8.0,
                    circle_point=True,
                )

    # ------------------------------------------------------------------
    # 3D Transform Gizmo Rendering & Hit-Testing
    # ------------------------------------------------------------------

    def _gizmo_length(self, pivot: np.ndarray) -> float:
        dist = float(np.linalg.norm(self.camera.eye() - pivot))
        return max(0.35, dist * 0.17)

    def _gizmo_axis_colors(self) -> Dict[str, Tuple[float, float, float, float]]:
        base = {
            "x": (0.94, 0.29, 0.31, 1.0),
            "y": (0.31, 0.84, 0.45, 1.0),
            "z": (0.29, 0.58, 0.96, 1.0),
            "free": (0.96, 0.96, 0.96, 0.95),
        }
        highlight = (1.0, 0.92, 0.35, 1.0)
        active = self._active_axis or self._hover_axis
        if active in base:
            base[active] = highlight
        return base

    def _draw_gizmo(self, pivot: np.ndarray, vp: np.ndarray) -> None:
        length = self._gizmo_length(pivot)
        colors = self._gizmo_axis_colors()
        axes = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

        tool = self.scene.tool

        if tool in (TransformTool.MOVE, TransformTool.SCALE):
            shaft_r = length * 0.022
            for key, axis_vec in axes.items():
                shaft_len = length * 0.82 if tool == TransformTool.MOVE else length * 0.94
                shaft_tris = self._build_cylinder_shaft(pivot, axis_vec, shaft_len, shaft_r, segs=8)
                self._draw_lines(shaft_tris, vp, colors[key], GL.GL_TRIANGLES)

                end = pivot + axis_vec * length
                if tool == TransformTool.MOVE:
                    tip_tris = self._build_cone_tip(
                        pivot + axis_vec * (length * 0.80), axis_vec, length * 0.24, length * 0.072
                    )
                    self._draw_lines(tip_tris, vp, colors[key], GL.GL_TRIANGLES)
                else:
                    self._draw_lines(
                        np.array([end], dtype=np.float32),
                        vp,
                        colors[key],
                        GL.GL_POINTS,
                        point_size=12.0,
                        circle_point=False,
                    )

            # Center free handle
            self._draw_lines(
                np.array([pivot], dtype=np.float32),
                vp,
                colors["free"],
                GL.GL_POINTS,
                point_size=12.0,
                circle_point=True,
            )

        elif tool == TransformTool.ROTATE:
            radius = length * 0.92
            band = length * 0.026
            for key, normal in axes.items():
                ring_tris = self._build_ring_ribbon(pivot, normal, radius, band, segments=48)
                self._draw_lines(ring_tris, vp, colors[key], GL.GL_TRIANGLES)

    @staticmethod
    def _ortho_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        u = normalize(np.cross(axis, ref))
        v = normalize(np.cross(axis, u))
        return u, v

    def _build_cylinder_shaft(
        self, start: np.ndarray, axis: np.ndarray, length: float, radius: float, segs: int = 8
    ) -> np.ndarray:
        u, v = self._ortho_basis(axis)
        end = start + axis * length
        tris: List[np.ndarray] = []
        for i in range(segs):
            a0 = 2.0 * math.pi * i / segs
            a1 = 2.0 * math.pi * (i + 1) / segs
            off0 = (u * math.cos(a0) + v * math.sin(a0)) * radius
            off1 = (u * math.cos(a1) + v * math.sin(a1)) * radius
            s0, s1 = start + off0, start + off1
            e0, e1 = end + off0, end + off1
            tris.extend([s0, s1, e1, s0, e1, e0])
        return np.asarray(tris, dtype=np.float32)

    def _build_cone_tip(
        self, base_center: np.ndarray, axis: np.ndarray, height: float, radius: float, segs: int = 12
    ) -> np.ndarray:
        u, v = self._ortho_basis(axis)
        apex = base_center + axis * height
        tris: List[np.ndarray] = []
        for i in range(segs):
            a0 = 2.0 * math.pi * i / segs
            a1 = 2.0 * math.pi * (i + 1) / segs
            p0 = base_center + (u * math.cos(a0) + v * math.sin(a0)) * radius
            p1 = base_center + (u * math.cos(a1) + v * math.sin(a1)) * radius
            tris.extend([apex, p0, p1])
        return np.asarray(tris, dtype=np.float32)

    def _build_ring_ribbon(
        self, center: np.ndarray, normal: np.ndarray, radius: float, half_width: float, segments: int = 48
    ) -> np.ndarray:
        u, v = self._ortho_basis(normal)
        r_tube = half_width * 0.85
        # 4-sided tube cross-section so the ring stays equally visible from every angle
        prof = [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
        ]
        rings: List[List[np.ndarray]] = []
        for i in range(segments):
            ang = 2.0 * math.pi * i / segments
            radial = u * math.cos(ang) + v * math.sin(ang)
            base = center + radial * radius
            rings.append([base + (radial * pr + normal * pn) * r_tube for pr, pn in prof])

        tris: List[np.ndarray] = []
        for i in range(segments):
            r0 = rings[i]
            r1 = rings[(i + 1) % segments]
            for k in range(4):
                kn = (k + 1) % 4
                tris.extend([r0[k], r1[k], r1[kn], r0[k], r1[kn], r0[kn]])
        return np.asarray(tris, dtype=np.float32)

    def _build_circle_loop(
        self, center: np.ndarray, normal: np.ndarray, radius: float, segments: int = 48
    ) -> np.ndarray:
        u, v = self._ortho_basis(normal)
        pts = []
        for i in range(segments):
            ang = 2.0 * math.pi * i / segments
            pts.append(center + (u * math.cos(ang) + v * math.sin(ang)) * radius)
        return np.asarray(pts, dtype=np.float32)

    def _hit_test_gizmo(self, mx: float, my: float) -> Optional[str]:
        pivot = self.scene.selection_pivot_world()
        if pivot is None or self.scene.tool == TransformTool.SELECT:
            return None

        w, h = self.width(), self.height()
        aspect = max(float(w), 1.0) / max(float(h), 1.0)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)
        length = self._gizmo_length(pivot)

        p0 = project_point(pivot, view, proj, w, h)
        if p0 is None:
            return None

        axes = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

        if self.scene.tool in (TransformTool.MOVE, TransformTool.SCALE):
            if math.hypot(mx - p0[0], my - p0[1]) <= 12.0:
                return "free"
            best_axis: Optional[str] = None
            best_dist = 13.0
            for key, axis_vec in axes.items():
                p1 = project_point(pivot + axis_vec * (length * 1.05), view, proj, w, h)
                if p1 is None:
                    continue
                d = distance_point_to_segment_2d(mx, my, p0[0], p0[1], p1[0], p1[1])
                if d < best_dist:
                    best_dist = d
                    best_axis = key
            return best_axis

        if self.scene.tool == TransformTool.ROTATE:
            radius = length * 0.92
            best_axis = None
            best_dist = 12.0
            for key, normal in axes.items():
                ring = self._build_circle_loop(pivot, normal, radius, segments=36)
                proj_pts = [project_point(pt, view, proj, w, h) for pt in ring]
                n = len(proj_pts)
                for i in range(n):
                    a = proj_pts[i]
                    b = proj_pts[(i + 1) % n]
                    if a is None or b is None:
                        continue
                    d = distance_point_to_segment_2d(mx, my, a[0], a[1], b[0], b[1])
                    if d < best_dist:
                        best_dist = d
                        best_axis = key
            return best_axis

        return None

    # ------------------------------------------------------------------
    # Mouse & Picking Logic
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.setFocus()
        pos = event.position()
        self._last_mouse = pos
        self._press_mouse = pos
        self._mouse_moved = False

        btn = event.button()
        mods = event.modifiers()

        # Camera navigation
        if btn == Qt.MouseButton.MiddleButton:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                self._drag_mode = "pan"
            else:
                self._drag_mode = "orbit"
            return
        if btn == Qt.MouseButton.RightButton:
            if mods & Qt.KeyboardModifier.AltModifier:
                self._drag_mode = "orbit"
            else:
                self._drag_mode = "pan"
            return
        if btn == Qt.MouseButton.LeftButton and (mods & Qt.KeyboardModifier.AltModifier):
            self._drag_mode = "orbit"
            return

        if btn == Qt.MouseButton.LeftButton:
            # Check top-right orientation triad click first
            if self._try_click_orientation_triad(pos.x(), pos.y()):
                return

            # Check gizmo hit
            axis = self._hit_test_gizmo(pos.x(), pos.y())
            if axis is not None:
                self._begin_gizmo_drag(axis, pos.x(), pos.y())
                return

            self._drag_mode = "pick"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        dx = pos.x() - self._last_mouse.x()
        dy = pos.y() - self._last_mouse.y()
        self._last_mouse = pos

        if math.hypot(pos.x() - self._press_mouse.x(), pos.y() - self._press_mouse.y()) > 3.0:
            self._mouse_moved = True

        if self._drag_mode == "orbit":
            self.camera.orbit(dx, dy)
            self.refresh()
            return

        if self._drag_mode == "pan":
            self.camera.pan(dx, dy, self.height())
            self.refresh()
            return

        if self._drag_mode == "gizmo" and self._active_axis is not None:
            self._update_gizmo_drag(pos.x(), pos.y())
            return

        # Idle hover feedback on gizmo
        new_hover = self._hit_test_gizmo(pos.x(), pos.y())
        if new_hover != self._hover_axis:
            self._hover_axis = new_hover
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if new_hover else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position()
        if self._drag_mode == "pick" and not self._mouse_moved:
            additive = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self._perform_pick(pos.x(), pos.y(), additive=additive)
        elif self._drag_mode == "gizmo":
            self._active_axis = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.scene_modified.emit()
            self.refresh()

        self._drag_mode = None

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y() / 120.0
        if abs(delta) > 1e-4:
            self.camera.zoom(delta)
            self.refresh()

    def _try_click_orientation_triad(self, mx: float, my: float) -> bool:
        cx, cy = self.width() - 58.0, 58.0
        radius = 34.0
        if math.hypot(mx - cx, my - cy) > radius + 16:
            return False
        rot = self.camera.view_matrix()[:3, :3]
        axes = [
            ("right", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("top", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("front", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]
        for preset, vec in axes:
            cam_v = rot @ vec
            ex = cx + float(cam_v[0]) * radius
            ey = cy - float(cam_v[1]) * radius
            if math.hypot(mx - ex, my - ey) <= 12.0:
                self.camera.set_preset(preset)
                self.status_message.emit(f"View: {preset.capitalize()}")
                self.refresh()
                return True
        return False

    # ------------------------------------------------------------------
    # Gizmo Drag Execution
    # ------------------------------------------------------------------

    def _axis_vector(self, axis: str) -> np.ndarray:
        mapping = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        return mapping.get(axis, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def _begin_gizmo_drag(self, axis: str, mx: float, my: float) -> None:
        pivot = self.scene.selection_pivot_world()
        if pivot is None:
            return

        self.scene.push_undo(f"{self.scene.tool.value.capitalize()} ({axis.upper()})")
        self._drag_mode = "gizmo"
        self._active_axis = axis
        self._gizmo_start_pivot = pivot.copy()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

        aspect = max(float(self.width()), 1.0) / max(float(self.height()), 1.0)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)
        ray_o, ray_d = screen_to_ray(mx, my, self.width(), self.height(), view, proj)

        if axis in ("x", "y", "z"):
            self._gizmo_start_param = closest_point_on_axis(
                ray_o, ray_d, pivot, self._axis_vector(axis)
            )
        else:
            hit = ray_plane_intersect(ray_o, ray_d, pivot, self.camera.forward())
            self._gizmo_start_plane_hit = hit if hit is not None else pivot.copy()

        p_screen = project_point(pivot, view, proj, self.width(), self.height())
        if p_screen is not None:
            self._gizmo_start_angle = math.atan2(my - p_screen[1], mx - p_screen[0])

        # Capture initial transforms or edit vertices
        self._gizmo_start_obj_transforms.clear()
        self._gizmo_start_edit_verts = None
        self._gizmo_edit_indices = []

        if self.scene.edit_mode == EditMode.OBJECT:
            for obj in self.scene.selected_objects():
                if not obj.locked:
                    self._gizmo_start_obj_transforms[obj.obj_id] = (
                        list(obj.position),
                        list(obj.rotation),
                        list(obj.scale),
                    )
        else:
            obj = self.scene.active_object()
            if obj is not None and not obj.locked:
                self._gizmo_edit_indices = sorted(self.scene.active_edit_vertex_indices())
                self._gizmo_start_edit_verts = obj.mesh.vertices.copy()

    def _snap_val(self, val: float) -> float:
        if not self.scene.snap_enabled or self.scene.snap_increment <= 1e-6:
            return val
        inc = self.scene.snap_increment
        return round(val / inc) * inc

    def _update_gizmo_drag(self, mx: float, my: float) -> None:
        axis = self._active_axis
        if axis is None:
            return
        pivot = self._gizmo_start_pivot
        aspect = max(float(self.width()), 1.0) / max(float(self.height()), 1.0)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)
        ray_o, ray_d = screen_to_ray(mx, my, self.width(), self.height(), view, proj)

        tool = self.scene.tool

        # Compute transform delta
        world_delta = np.zeros(3, dtype=np.float32)
        rot_deg = np.zeros(3, dtype=np.float32)
        scale_factors = np.ones(3, dtype=np.float32)

        if tool == TransformTool.MOVE:
            if axis in ("x", "y", "z"):
                vec = self._axis_vector(axis)
                curr_s = closest_point_on_axis(ray_o, ray_d, pivot, vec)
                ds = self._snap_val(curr_s - self._gizmo_start_param)
                world_delta = vec * ds
            else:
                hit = ray_plane_intersect(ray_o, ray_d, pivot, self.camera.forward())
                if hit is not None:
                    d = hit - self._gizmo_start_plane_hit
                    world_delta = np.array([self._snap_val(float(c)) for c in d], dtype=np.float32)

        elif tool == TransformTool.ROTATE:
            p_screen = project_point(pivot, view, proj, self.width(), self.height())
            if p_screen is not None:
                ang = math.atan2(my - p_screen[1], mx - p_screen[0])
                delta_deg = math.degrees(ang - self._gizmo_start_angle)
                if self.scene.snap_enabled:
                    delta_deg = round(delta_deg / 15.0) * 15.0
                idx = {"x": 0, "y": 1, "z": 2}.get(axis, 1)
                rot_deg[idx] = -delta_deg

        elif tool == TransformTool.SCALE:
            if axis in ("x", "y", "z"):
                vec = self._axis_vector(axis)
                curr_s = closest_point_on_axis(ray_o, ray_d, pivot, vec)
                ds = curr_s - self._gizmo_start_param
                factor = max(0.05, 1.0 + ds * 0.75)
                idx = {"x": 0, "y": 1, "z": 2}[axis]
                scale_factors[idx] = factor
            else:
                dx = (mx - self._press_mouse.x()) * 0.01
                factor = max(0.05, 1.0 + dx)
                scale_factors[:] = factor

        # Apply in Object Mode
        if self.scene.edit_mode == EditMode.OBJECT:
            for obj_id, (p0, r0, s0) in self._gizmo_start_obj_transforms.items():
                obj = self.scene.get_object(obj_id)
                if obj is None:
                    continue
                if tool == TransformTool.MOVE:
                    obj.position = [p0[0] + float(world_delta[0]), p0[1] + float(world_delta[1]), p0[2] + float(world_delta[2])]
                elif tool == TransformTool.ROTATE:
                    obj.rotation = [r0[0] + float(rot_deg[0]), r0[1] + float(rot_deg[1]), r0[2] + float(rot_deg[2])]
                elif tool == TransformTool.SCALE:
                    obj.scale = [s0[0] * float(scale_factors[0]), s0[1] * float(scale_factors[1]), s0[2] * float(scale_factors[2])]
        else:
            # Apply in Edit Mode (modify vertices in local space)
            obj = self.scene.active_object()
            if obj is not None and self._gizmo_start_edit_verts is not None and self._gizmo_edit_indices:
                model = obj.model_matrix()
                inv_model = np.linalg.inv(model.astype(np.float64)).astype(np.float32)
                inv_rot_scale = inv_model[:3, :3]

                verts = self._gizmo_start_edit_verts.copy()
                local_pivot = (inv_model @ np.append(pivot, 1.0))[:3]

                if tool == TransformTool.MOVE:
                    local_delta = (inv_rot_scale @ world_delta).astype(np.float32)
                    verts[self._gizmo_edit_indices] += local_delta
                elif tool == TransformTool.ROTATE:
                    rmat = rotate_euler_xyz(float(rot_deg[0]), float(rot_deg[1]), float(rot_deg[2]))[:3, :3]
                    rel = verts[self._gizmo_edit_indices] - local_pivot
                    verts[self._gizmo_edit_indices] = (rel @ rmat.T) + local_pivot
                elif tool == TransformTool.SCALE:
                    rel = verts[self._gizmo_edit_indices] - local_pivot
                    verts[self._gizmo_edit_indices] = rel * scale_factors + local_pivot

                obj.mesh.vertices = verts
                obj.mesh.touch()

        self.scene_modified.emit()
        self.refresh()

    # ------------------------------------------------------------------
    # Ray / Screen Picking
    # ------------------------------------------------------------------

    def _perform_pick(self, mx: float, my: float, additive: bool) -> None:
        aspect = max(float(self.width()), 1.0) / max(float(self.height()), 1.0)
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix(aspect)
        ray_o, ray_d = screen_to_ray(mx, my, self.width(), self.height(), view, proj)

        if self.scene.edit_mode == EditMode.OBJECT:
            best_t = float("inf")
            best_id: Optional[str] = None
            for obj in self.scene.objects:
                if not obj.visible or obj.locked:
                    continue
                bmin, bmax = obj.world_bounds()
                if ray_aabb_intersect(ray_o, ray_d, bmin, bmax) is None:
                    continue
                wv = obj.world_vertices()
                for face in obj.mesh.faces:
                    if len(face) < 3:
                        continue
                    v0 = wv[face[0]]
                    for k in range(1, len(face) - 1):
                        t = ray_triangle_intersect(ray_o, ray_d, v0, wv[face[k]], wv[face[k + 1]])
                        if t is not None and t < best_t:
                            best_t = t
                            best_id = obj.obj_id
            self.scene.select_object(best_id, additive=additive)
            self.selection_changed.emit()
            self.refresh()
            return

        # Edit Mode picking on active object
        obj = self.scene.active_object()
        if obj is None or not obj.visible:
            return
        wv = obj.world_vertices()
        w, h = self.width(), self.height()

        if self.scene.edit_mode == EditMode.FACE:
            best_t = float("inf")
            best_face: Optional[int] = None
            for fi, face in enumerate(obj.mesh.faces):
                if len(face) < 3:
                    continue
                v0 = wv[face[0]]
                for k in range(1, len(face) - 1):
                    t = ray_triangle_intersect(ray_o, ray_d, v0, wv[face[k]], wv[face[k + 1]])
                    if t is not None and t < best_t:
                        best_t = t
                        best_face = fi
            if not additive:
                self.scene.selected_faces.clear()
            if best_face is not None:
                if additive and best_face in self.scene.selected_faces:
                    self.scene.selected_faces.remove(best_face)
                else:
                    self.scene.selected_faces.add(best_face)

        elif self.scene.edit_mode == EditMode.VERTEX:
            best_vi: Optional[int] = None
            best_score = float("inf")
            for vi, pt in enumerate(wv):
                proj_p = project_point(pt, view, proj, w, h)
                if proj_p is None:
                    continue
                dist = math.hypot(mx - proj_p[0], my - proj_p[1])
                if dist <= 14.0:
                    # Prefer closer depth when multiple vertices overlap
                    score = dist + proj_p[2] * 3.0
                    if score < best_score:
                        best_score = score
                        best_vi = vi
            if not additive:
                self.scene.selected_vertices.clear()
            if best_vi is not None:
                if additive and best_vi in self.scene.selected_vertices:
                    self.scene.selected_vertices.remove(best_vi)
                else:
                    self.scene.selected_vertices.add(best_vi)

        elif self.scene.edit_mode == EditMode.EDGE:
            best_edge: Optional[Edge] = None
            best_dist = 11.0
            for edge in obj.mesh.unique_edges():
                p0 = project_point(wv[edge[0]], view, proj, w, h)
                p1 = project_point(wv[edge[1]], view, proj, w, h)
                if p0 is None or p1 is None:
                    continue
                d = distance_point_to_segment_2d(mx, my, p0[0], p0[1], p1[0], p1[1])
                if d < best_dist:
                    best_dist = d
                    best_edge = edge
            if not additive:
                self.scene.selected_edges.clear()
            if best_edge is not None:
                if additive and best_edge in self.scene.selected_edges:
                    self.scene.selected_edges.remove(best_edge)
                else:
                    self.scene.selected_edges.add(best_edge)

        self.selection_changed.emit()
        self.refresh()

    # ------------------------------------------------------------------
    # Camera Framing Helpers
    # ------------------------------------------------------------------

    def frame_selection(self) -> None:
        selected = self.scene.selected_objects()
        targets = selected if selected else [o for o in self.scene.objects if o.visible]
        if not targets:
            self.camera.reset()
            self.refresh()
            return
        mins = []
        maxs = []
        for obj in targets:
            bmin, bmax = obj.world_bounds()
            mins.append(bmin)
            maxs.append(bmax)
        all_min = np.min(mins, axis=0)
        all_max = np.max(maxs, axis=0)
        self.camera.frame_bounds(all_min, all_max)
        self.refresh()
