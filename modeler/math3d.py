"""3D linear algebra, projection matrices, and ray-geometry intersection utilities."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Return unit vector along v, or zero vector if degenerate."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.zeros_like(v, dtype=np.float32)
    return (v / norm).astype(np.float32)


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create a right-handed OpenGL perspective projection matrix (4x4 float32)."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    aspect = max(aspect, 1e-4)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Create an OpenGL orthographic projection matrix (4x4 float32)."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a right-handed OpenGL look-at view matrix (4x4 float32)."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    if np.linalg.norm(s) < 1e-6:
        # Handle looking straight up or down
        alt_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        s = normalize(np.cross(f, alt_up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0:3] = s
    m[1, 0:3] = u
    m[2, 0:3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def translate(tx: float, ty: float, tz: float) -> np.ndarray:
    """Create a 4x4 translation matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """Create a 4x4 scaling matrix."""
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotate_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4, dtype=np.float32)
    m[1, 1], m[1, 2] = c, -s
    m[2, 1], m[2, 2] = s, c
    return m


def rotate_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m


def rotate_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1] = c, -s
    m[1, 0], m[1, 1] = s, c
    return m


def rotate_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler rotation matrix applying X, then Y, then Z (Rz @ Ry @ Rx)."""
    return (rotate_z(rz) @ rotate_y(ry) @ rotate_x(rx)).astype(np.float32)


def trs_matrix(
    position: Tuple[float, float, float] | np.ndarray,
    rotation: Tuple[float, float, float] | np.ndarray,
    scale: Tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Compose Translation * Rotation * Scale 4x4 matrix."""
    t = translate(float(position[0]), float(position[1]), float(position[2]))
    r = rotate_euler_xyz(float(rotation[0]), float(rotation[1]), float(rotation[2]))
    s = scale_matrix(float(scale[0]), float(scale[1]), float(scale[2]))
    return (t @ r @ s).astype(np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points by a 4x4 matrix."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    rot_scale = matrix[:3, :3]
    trans = matrix[:3, 3]
    return (pts @ rot_scale.T + trans).astype(np.float32)


def screen_to_ray(
    x: float,
    y: float,
    width: int,
    height: int,
    view: np.ndarray,
    proj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject screen coordinates (x, y top-left origin) to a world-space ray (origin, dir)."""
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    ndc_x = (2.0 * x) / w - 1.0
    ndc_y = 1.0 - (2.0 * y) / h

    inv_vp = np.linalg.inv((proj @ view).astype(np.float64))

    near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float64)
    far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float64)

    near_world = inv_vp @ near_ndc
    far_world = inv_vp @ far_ndc
    if abs(near_world[3]) > 1e-10:
        near_world /= near_world[3]
    if abs(far_world[3]) > 1e-10:
        far_world /= far_world[3]

    origin = near_world[:3].astype(np.float32)
    direction = normalize((far_world[:3] - near_world[:3]).astype(np.float32))
    return origin, direction


def project_point(
    point: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float]]:
    """Project a 3D world point to screen space (x, y, depth_ndc). Returns None if behind camera."""
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    clip = (proj @ view).astype(np.float64) @ p
    if clip[3] <= 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] * 0.5 + 0.5) * width
    sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
    return float(sx), float(sy), float(ndc[2])


def ray_aabb_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> Optional[float]:
    """Slab method ray-AABB intersection. Returns entry t >= 0 or None."""
    tmin = -float("inf")
    tmax = float("inf")
    for i in range(3):
        d = float(direction[i])
        if abs(d) < 1e-8:
            if origin[i] < bmin[i] or origin[i] > bmax[i]:
                return None
        else:
            inv_d = 1.0 / d
            t1 = (bmin[i] - origin[i]) * inv_d
            t2 = (bmax[i] - origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    if tmax < 0.0:
        return None
    return max(tmin, 0.0)


def ray_triangle_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> Optional[float]:
    """Moller-Trumbore ray-triangle intersection. Returns t > 0 or None."""
    eps = 1e-7
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(direction, e2)
    a = float(np.dot(e1, h))
    if -eps < a < eps:
        return None
    f = 1.0 / a
    s = origin - v0
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, e1)
    v = f * float(np.dot(direction, q))
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * float(np.dot(e2, q))
    if t > eps:
        return t
    return None


def ray_plane_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[np.ndarray]:
    """Intersect a ray with a plane. Returns world hit point or None if parallel."""
    denom = float(np.dot(plane_normal, direction))
    if abs(denom) < 1e-6:
        return None
    t = float(np.dot(plane_point - origin, plane_normal)) / denom
    if t < 0.0:
        return None
    return (origin + direction * t).astype(np.float32)


def closest_point_on_axis(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    axis_origin: np.ndarray,
    axis_dir: np.ndarray,
) -> float:
    """Return scalar parameter `s` along axis_origin + s * axis_dir closest to the ray."""
    w0 = axis_origin - ray_origin
    a = float(np.dot(axis_dir, axis_dir))
    b = float(np.dot(axis_dir, ray_dir))
    c = float(np.dot(ray_dir, ray_dir))
    d = float(np.dot(axis_dir, w0))
    e = float(np.dot(ray_dir, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-7:
        return 0.0
    return (b * e - c * d) / denom


def distance_point_to_segment_2d(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """2D Euclidean distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-8:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)
