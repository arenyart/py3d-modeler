"""Procedural 3D primitive mesh generators."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .mesh import Mesh


def create_cube(size: float = 1.6) -> Mesh:
    """Create a centered quad cube with outward counter-clockwise face normals."""
    h = size * 0.5
    verts = [
        [-h, -h,  h],  # 0 front-bottom-left
        [ h, -h,  h],  # 1 front-bottom-right
        [ h,  h,  h],  # 2 front-top-right
        [-h,  h,  h],  # 3 front-top-left
        [-h, -h, -h],  # 4 back-bottom-left
        [ h, -h, -h],  # 5 back-bottom-right
        [ h,  h, -h],  # 6 back-top-right
        [-h,  h, -h],  # 7 back-top-left
    ]
    faces = [
        [0, 1, 2, 3],  # +Z front
        [5, 4, 7, 6],  # -Z back
        [4, 0, 3, 7],  # -X left
        [1, 5, 6, 2],  # +X right
        [3, 2, 6, 7],  # +Y top
        [4, 5, 1, 0],  # -Y bottom
    ]
    return Mesh(verts, faces)


def create_plane(size: float = 4.0, subdivisions: int = 4) -> Mesh:
    """Create a subdivided horizontal XZ plane centered at origin."""
    n = max(1, int(subdivisions))
    half = size * 0.5
    verts: List[List[float]] = []
    for iz in range(n + 1):
        z = -half + (iz / n) * size
        for ix in range(n + 1):
            x = -half + (ix / n) * size
            verts.append([x, 0.0, z])

    faces: List[List[int]] = []
    stride = n + 1
    for iz in range(n):
        for ix in range(n):
            v0 = iz * stride + ix
            v1 = v0 + 1
            v2 = v0 + stride + 1
            v3 = v0 + stride
            # Outward normal +Y: (v3 - v0) x (v1 - v0) = (0,0,+) x (+,0,0) = (0,+,0)
            faces.append([v0, v3, v2, v1])
    return Mesh(verts, faces)


def create_uv_sphere(radius: float = 1.0, rings: int = 16, segments: int = 24) -> Mesh:
    """Create a UV sphere with triangle caps and quad bands."""
    rings = max(4, int(rings))
    segments = max(6, int(segments))

    verts: List[List[float]] = [[0.0, radius, 0.0]]  # 0: north pole
    for r in range(1, rings):
        phi = math.pi * r / rings
        y = radius * math.cos(phi)
        ring_r = radius * math.sin(phi)
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments
            x = ring_r * math.cos(theta)
            z = ring_r * math.sin(theta)
            verts.append([x, y, z])
    south_idx = len(verts)
    verts.append([0.0, -radius, 0.0])

    faces: List[List[int]] = []
    # Top cap
    for s in range(segments):
        s_next = (s + 1) % segments
        faces.append([0, 1 + s_next, 1 + s])

    # Middle quad rings
    for r in range(rings - 2):
        row0 = 1 + r * segments
        row1 = 1 + (r + 1) * segments
        for s in range(segments):
            s_next = (s + 1) % segments
            faces.append([row0 + s, row0 + s_next, row1 + s_next, row1 + s])

    # Bottom cap
    last_row = 1 + (rings - 2) * segments
    for s in range(segments):
        s_next = (s + 1) % segments
        faces.append([south_idx, last_row + s, last_row + s_next])

    return Mesh(verts, faces)


def create_icosphere(radius: float = 1.0, subdivisions: int = 2) -> Mesh:
    """Create a geodesic icosphere by subdividing an icosahedron."""
    t = (1.0 + math.sqrt(5.0)) * 0.5
    raw_verts = [
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ]
    verts = [
        (np.array(v, dtype=np.float32) / float(np.linalg.norm(v)) * radius)
        for v in raw_verts
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    for _ in range(max(0, min(4, int(subdivisions)))):
        mid_cache: Dict[Tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key in mid_cache:
                return mid_cache[key]
            m = (verts[a] + verts[b]) * 0.5
            m = m / float(np.linalg.norm(m)) * radius
            idx = len(verts)
            verts.append(m.astype(np.float32))
            mid_cache[key] = idx
            return idx

        new_faces: List[List[int]] = []
        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        faces = new_faces

    return Mesh(verts, faces)


def create_cylinder(radius: float = 0.8, height: float = 1.8, segments: int = 24) -> Mesh:
    """Create a capped cylinder centered at origin along the Y axis."""
    segments = max(3, int(segments))
    half_h = height * 0.5

    verts: List[List[float]] = []
    # Top ring: 0 .. segments-1
    for s in range(segments):
        theta = 2.0 * math.pi * s / segments
        verts.append([radius * math.cos(theta), half_h, radius * math.sin(theta)])
    # Bottom ring: segments .. 2*segments-1
    for s in range(segments):
        theta = 2.0 * math.pi * s / segments
        verts.append([radius * math.cos(theta), -half_h, radius * math.sin(theta)])

    faces: List[List[int]] = []
    # Side quads (outward normal)
    for s in range(segments):
        s_next = (s + 1) % segments
        t0, t1 = s, s_next
        b0, b1 = segments + s, segments + s_next
        faces.append([t0, t1, b1, b0])

    # Top cap (+Y normal: reverse order of XZ circle)
    faces.append(list(reversed(range(segments))))
    # Bottom cap (-Y normal)
    faces.append(list(range(segments, 2 * segments)))

    return Mesh(verts, faces)


def create_cone(radius: float = 0.9, height: float = 1.8, segments: int = 24) -> Mesh:
    """Create a capped cone centered at origin along the Y axis."""
    segments = max(3, int(segments))
    half_h = height * 0.5

    verts: List[List[float]] = [[0.0, half_h, 0.0]]  # 0: apex
    for s in range(segments):
        theta = 2.0 * math.pi * s / segments
        verts.append([radius * math.cos(theta), -half_h, radius * math.sin(theta)])

    faces: List[List[int]] = []
    for s in range(segments):
        s_next = (s + 1) % segments
        faces.append([0, 1 + s_next, 1 + s])

    # Base cap (-Y normal)
    faces.append(list(range(1, 1 + segments)))
    return Mesh(verts, faces)


def create_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
    major_segments: int = 28,
    minor_segments: int = 16,
) -> Mesh:
    """Create a torus lying in the XZ plane."""
    maj = max(6, int(major_segments))
    min_s = max(4, int(minor_segments))

    verts: List[List[float]] = []
    for i in range(maj):
        u = 2.0 * math.pi * i / maj
        cu, su = math.cos(u), math.sin(u)
        for j in range(min_s):
            v = 2.0 * math.pi * j / min_s
            cv, sv = math.cos(v), math.sin(v)
            r = major_radius + minor_radius * cv
            x = r * cu
            y = minor_radius * sv
            z = r * su
            verts.append([x, y, z])

    faces: List[List[int]] = []
    for i in range(maj):
        i_next = (i + 1) % maj
        for j in range(min_s):
            j_next = (j + 1) % min_s
            v00 = i * min_s + j
            v10 = i_next * min_s + j
            v11 = i_next * min_s + j_next
            v01 = i * min_s + j_next
            faces.append([v00, v01, v11, v10])

    return Mesh(verts, faces)


def create_pyramid(base_size: float = 1.6, height: float = 1.6) -> Mesh:
    """Create a 4-sided pyramid centered at origin."""
    h = base_size * 0.5
    hy = height * 0.5
    verts = [
        [ 0.0,  hy,  0.0],  # 0 apex
        [-h,   -hy,  h],    # 1 front-left
        [ h,   -hy,  h],    # 2 front-right
        [ h,   -hy, -h],    # 3 back-right
        [-h,   -hy, -h],    # 4 back-left
    ]
    faces = [
        [0, 1, 2],  # front
        [0, 2, 3],  # right
        [0, 3, 4],  # back
        [0, 4, 1],  # left
        [4, 3, 2, 1],  # bottom (-Y)
    ]
    return Mesh(verts, faces)


PRIMITIVE_BUILDERS = {
    "Cube": create_cube,
    "UV Sphere": create_uv_sphere,
    "Icosphere": create_icosphere,
    "Cylinder": create_cylinder,
    "Cone": create_cone,
    "Torus": create_torus,
    "Plane": create_plane,
    "Pyramid": create_pyramid,
}
