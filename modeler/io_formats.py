"""Import and export 3D mesh formats: Wavefront OBJ, STL (ASCII & Binary), and PLY."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mesh import Mesh
from .scene import SceneObject


def export_obj(path: str | Path, objects: Sequence[SceneObject]) -> None:
    """Export one or more SceneObjects in world space to a Wavefront .obj file."""
    lines: List[str] = [
        "# Exported by Py3D Modeler",
        f"# Objects: {len(objects)}",
        "",
    ]
    v_offset = 1
    for obj in objects:
        safe_name = obj.name.replace(" ", "_")
        lines.append(f"o {safe_name}")
        world_v = obj.world_vertices()
        for v in world_v:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for face in obj.mesh.faces:
            idx_str = " ".join(str(i + v_offset) for i in face)
            lines.append(f"f {idx_str}")
        lines.append("")
        v_offset += len(world_v)

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def import_obj(path: str | Path) -> List[Tuple[str, Mesh]]:
    """Import a Wavefront .obj file, returning a list of (name, Mesh) pairs."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    all_verts: List[List[float]] = []
    groups: List[Tuple[str, List[List[int]]]] = []
    current_name = Path(path).stem
    current_faces: List[List[int]] = []

    def flush_group() -> None:
        nonlocal current_faces
        if current_faces:
            groups.append((current_name, current_faces))
            current_faces = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cmd = parts[0]
        if cmd == "v" and len(parts) >= 4:
            all_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif cmd in ("o", "g") and len(parts) >= 2:
            flush_group()
            current_name = " ".join(parts[1:])
        elif cmd == "f" and len(parts) >= 4:
            face: List[int] = []
            for tok in parts[1:]:
                idx_str = tok.split("/")[0]
                if not idx_str:
                    continue
                idx = int(idx_str)
                if idx < 0:
                    idx = len(all_verts) + idx
                else:
                    idx -= 1
                if 0 <= idx < len(all_verts):
                    face.append(idx)
            if len(face) >= 3:
                current_faces.append(face)

    flush_group()

    if not groups:
        return []

    results: List[Tuple[str, Mesh]] = []
    verts_arr = np.asarray(all_verts, dtype=np.float32)
    for name, faces in groups:
        used = sorted({vi for f in faces for vi in f})
        remap = {old: new for new, old in enumerate(used)}
        sub_verts = verts_arr[used]
        sub_faces = [[remap[vi] for vi in f] for f in faces]
        results.append((name, Mesh(sub_verts, sub_faces)))
    return results


def export_stl_binary(path: str | Path, objects: Sequence[SceneObject]) -> None:
    """Export one or more SceneObjects in world space to binary STL."""
    tris: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for obj in objects:
        world_v = obj.world_vertices()
        for face in obj.mesh.faces:
            if len(face) < 3:
                continue
            for k in range(1, len(face) - 1):
                v0 = world_v[face[0]]
                v1 = world_v[face[k]]
                v2 = world_v[face[k + 1]]
                n = np.cross(v1 - v0, v2 - v0)
                norm = float(np.linalg.norm(n))
                n = (n / norm) if norm > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
                tris.append((n, v0, v1, v2))

    header = b"Exported by Py3D Modeler".ljust(80, b"\x00")[:80]
    with open(path, "wb") as fp:
        fp.write(header)
        fp.write(struct.pack("<I", len(tris)))
        for n, v0, v1, v2 in tris:
            fp.write(
                struct.pack(
                    "<12fH",
                    float(n[0]), float(n[1]), float(n[2]),
                    float(v0[0]), float(v0[1]), float(v0[2]),
                    float(v1[0]), float(v1[1]), float(v1[2]),
                    float(v2[0]), float(v2[1]), float(v2[2]),
                    0,
                )
            )


def import_stl(path: str | Path) -> Mesh:
    """Import an ASCII or Binary STL file with automatic vertex welding."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        return _import_stl_ascii(raw.decode("utf-8", errors="ignore"))

    tri_count = struct.unpack("<I", raw[80:84])[0]
    expected_size = 84 + tri_count * 50
    if expected_size == len(raw):
        return _import_stl_binary(raw, tri_count)
    return _import_stl_ascii(raw.decode("utf-8", errors="ignore"))


def _weld_triangles(tri_verts: List[Tuple[float, float, float]]) -> Mesh:
    vert_map: Dict[Tuple[float, float, float], int] = {}
    verts: List[List[float]] = []
    faces: List[List[int]] = []

    def get_idx(pt: Tuple[float, float, float]) -> int:
        key = (round(pt[0], 5), round(pt[1], 5), round(pt[2], 5))
        if key not in vert_map:
            vert_map[key] = len(verts)
            verts.append([pt[0], pt[1], pt[2]])
        return vert_map[key]

    for i in range(0, len(tri_verts) - 2, 3):
        i0 = get_idx(tri_verts[i])
        i1 = get_idx(tri_verts[i + 1])
        i2 = get_idx(tri_verts[i + 2])
        if len({i0, i1, i2}) == 3:
            faces.append([i0, i1, i2])
    return Mesh(verts, faces)


def _import_stl_binary(raw: bytes, tri_count: int) -> Mesh:
    tri_verts: List[Tuple[float, float, float]] = []
    offset = 84
    for _ in range(tri_count):
        vals = struct.unpack_from("<12fH", raw, offset)
        offset += 50
        tri_verts.append((vals[3], vals[4], vals[5]))
        tri_verts.append((vals[6], vals[7], vals[8]))
        tri_verts.append((vals[9], vals[10], vals[11]))
    return _weld_triangles(tri_verts)


def _import_stl_ascii(text: str) -> Mesh:
    tri_verts: List[Tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            tri_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return _weld_triangles(tri_verts)


def export_ply(path: str | Path, objects: Sequence[SceneObject]) -> None:
    """Export one or more SceneObjects to ASCII Stanford PLY."""
    all_verts: List[np.ndarray] = []
    all_faces: List[List[int]] = []
    offset = 0
    for obj in objects:
        wv = obj.world_vertices()
        all_verts.extend(wv)
        for f in obj.mesh.faces:
            all_faces.append([vi + offset for vi in f])
        offset += len(wv)

    lines = [
        "ply",
        "format ascii 1.0",
        "comment Exported by Py3D Modeler",
        f"element vertex {len(all_verts)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(all_faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for v in all_verts:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in all_faces:
        lines.append(f"{len(f)} " + " ".join(str(i) for i in f))

    Path(path).write_text("\n".join(lines), encoding="utf-8")
