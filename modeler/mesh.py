"""Polygonal 3D mesh representation and modeling operations."""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .math3d import normalize


Edge = Tuple[int, int]


def canonical_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


class Mesh:
    """Polygonal mesh supporting triangles, quads, and n-gons."""

    def __init__(
        self,
        vertices: Sequence[Sequence[float]] | np.ndarray,
        faces: Sequence[Sequence[int]],
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=np.float32)
        if self.vertices.size == 0:
            self.vertices = np.zeros((0, 3), dtype=np.float32)
        else:
            self.vertices = self.vertices.reshape(-1, 3)
        self.faces: List[List[int]] = [list(f) for f in faces if len(f) >= 3]
        self._version: int = 0

    def clone(self) -> "Mesh":
        m = Mesh(self.vertices.copy(), [list(f) for f in self.faces])
        return m

    def touch(self) -> None:
        self._version += 1

    @property
    def version(self) -> int:
        return self._version

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def triangle_count(self) -> int:
        return sum(max(0, len(f) - 2) for f in self.faces)

    def unique_edges(self) -> List[Edge]:
        """Return sorted list of unique undirected edges (u, v) with u < v."""
        edge_set: Set[Edge] = set()
        for face in self.faces:
            n = len(face)
            for i in range(n):
                u, v = face[i], face[(i + 1) % n]
                if u != v:
                    edge_set.add(canonical_edge(u, v))
        return sorted(edge_set)

    @property
    def edge_count(self) -> int:
        return len(self.unique_edges())

    def face_normal(self, face_idx: int) -> np.ndarray:
        """Compute unit normal for a single polygon using Newell's method."""
        face = self.faces[face_idx]
        if len(face) < 3:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        pts = self.vertices[face]
        # Newell's method handles non-planar quads & n-gons robustly
        n = np.zeros(3, dtype=np.float32)
        count = len(pts)
        for i in range(count):
            p1 = pts[i]
            p2 = pts[(i + 1) % count]
            n[0] += (p1[1] - p2[1]) * (p1[2] + p2[2])
            n[1] += (p1[2] - p2[2]) * (p1[0] + p2[0])
            n[2] += (p1[0] - p2[0]) * (p1[1] + p2[1])
        out = normalize(n)
        if np.linalg.norm(out) < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return out

    def face_center(self, face_idx: int) -> np.ndarray:
        face = self.faces[face_idx]
        if not face:
            return np.zeros(3, dtype=np.float32)
        return self.vertices[face].mean(axis=0).astype(np.float32)

    def compute_face_normals(self) -> np.ndarray:
        if not self.faces:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([self.face_normal(i) for i in range(len(self.faces))], dtype=np.float32)

    def compute_vertex_normals(self) -> np.ndarray:
        """Area-weighted smooth vertex normals."""
        v_normals = np.zeros_like(self.vertices, dtype=np.float32)
        for face in self.faces:
            if len(face) < 3:
                continue
            p0 = self.vertices[face[0]]
            for i in range(1, len(face) - 1):
                p1 = self.vertices[face[i]]
                p2 = self.vertices[face[i + 1]]
                cross = np.cross(p1 - p0, p2 - p0)
                v_normals[face[0]] += cross
                v_normals[face[i]] += cross
                v_normals[face[i + 1]] += cross
        norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (v_normals / norms).astype(np.float32)

    def triangulated_arrays(
        self, smooth: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions, normals, face_ids) expanded per triangle vertex for GPU rendering.

        Returns:
            positions: (T*3, 3) float32
            normals: (T*3, 3) float32
            face_ids: (T*3,) int32 mapping each triangle vertex back to its source polygon index
        """
        if not self.faces or len(self.vertices) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        v_normals = self.compute_vertex_normals() if smooth else None
        positions: List[np.ndarray] = []
        normals: List[np.ndarray] = []
        face_ids: List[int] = []

        for f_idx, face in enumerate(self.faces):
            if len(face) < 3:
                continue
            fn = None if smooth else self.face_normal(f_idx)
            i0 = face[0]
            for k in range(1, len(face) - 1):
                i1 = face[k]
                i2 = face[k + 1]
                positions.append(self.vertices[i0])
                positions.append(self.vertices[i1])
                positions.append(self.vertices[i2])
                if smooth and v_normals is not None:
                    normals.append(v_normals[i0])
                    normals.append(v_normals[i1])
                    normals.append(v_normals[i2])
                else:
                    assert fn is not None
                    normals.append(fn)
                    normals.append(fn)
                    normals.append(fn)
                face_ids.extend((f_idx, f_idx, f_idx))

        if not positions:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        return (
            np.asarray(positions, dtype=np.float32),
            np.asarray(normals, dtype=np.float32),
            np.asarray(face_ids, dtype=np.int32),
        )

    def edge_positions(self) -> np.ndarray:
        """Return (E*2, 3) float32 array of line segment endpoints for wireframe rendering."""
        edges = self.unique_edges()
        if not edges or len(self.vertices) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        idx = np.asarray(edges, dtype=np.int32).reshape(-1)
        return self.vertices[idx].astype(np.float32)

    def compute_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return local-space (min_xyz, max_xyz)."""
        if len(self.vertices) == 0:
            z = np.zeros(3, dtype=np.float32)
            return z, z.copy()
        return self.vertices.min(axis=0).astype(np.float32), self.vertices.max(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Modeling / Mesh Editing Operations
    # ------------------------------------------------------------------

    def center_geometry(self) -> np.ndarray:
        """Center vertices around the origin and return the old centroid offset."""
        if len(self.vertices) == 0:
            return np.zeros(3, dtype=np.float32)
        bmin, bmax = self.compute_bounds()
        center = ((bmin + bmax) * 0.5).astype(np.float32)
        self.vertices = (self.vertices - center).astype(np.float32)
        self.touch()
        return center

    def flip_normals(self, face_indices: Optional[Iterable[int]] = None) -> None:
        """Reverse winding order of specified faces (or all faces if None)."""
        indices = range(len(self.faces)) if face_indices is None else face_indices
        for idx in indices:
            if 0 <= idx < len(self.faces):
                self.faces[idx] = list(reversed(self.faces[idx]))
        self.touch()

    def triangulate(self, face_indices: Optional[Iterable[int]] = None) -> None:
        """Convert quads and n-gons into triangles."""
        target = None if face_indices is None else set(face_indices)
        new_faces: List[List[int]] = []
        for idx, face in enumerate(self.faces):
            if (target is None or idx in target) and len(face) > 3:
                for k in range(1, len(face) - 1):
                    new_faces.append([face[0], face[k], face[k + 1]])
            else:
                new_faces.append(list(face))
        self.faces = new_faces
        self.touch()

    def delete_faces(self, face_indices: Iterable[int]) -> None:
        """Delete faces by index and prune unreferenced vertices."""
        to_delete = set(face_indices)
        if not to_delete:
            return
        self.faces = [f for i, f in enumerate(self.faces) if i not in to_delete]
        self._prune_unused_vertices()
        self.touch()

    def delete_vertices(self, vert_indices: Iterable[int]) -> None:
        """Delete vertices and any incident faces."""
        to_delete = set(vert_indices)
        if not to_delete:
            return
        self.faces = [f for f in self.faces if not any(v in to_delete for v in f)]
        self._prune_unused_vertices()
        self.touch()

    def _prune_unused_vertices(self) -> None:
        used = sorted({v for f in self.faces for v in f})
        if not used:
            self.vertices = np.zeros((0, 3), dtype=np.float32)
            self.faces = []
            return
        remap = {old: new for new, old in enumerate(used)}
        self.vertices = self.vertices[used].astype(np.float32)
        self.faces = [[remap[v] for v in f] for f in self.faces]

    def extrude_faces(self, face_indices: Sequence[int], distance: float = 0.5) -> List[int]:
        """Extrude selected faces as a connected region along their average normal.

        Returns the list of indices of the newly extruded cap faces.
        """
        selected = sorted({i for i in face_indices if 0 <= i < len(self.faces)})
        if not selected:
            return []

        selected_set = set(selected)

        # Average normal of the selected region
        avg_normal = np.zeros(3, dtype=np.float32)
        for fi in selected:
            avg_normal += self.face_normal(fi)
        avg_normal = normalize(avg_normal)
        if np.linalg.norm(avg_normal) < 1e-6:
            avg_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Count directed boundary edges within the selected face set
        directed_edges: Set[Tuple[int, int]] = set()
        for fi in selected:
            face = self.faces[fi]
            n = len(face)
            for k in range(n):
                directed_edges.add((face[k], face[(k + 1) % n]))

        # A directed edge (u, v) is on the boundary of the extruded region if (v, u) is not also selected
        boundary_edges: List[Tuple[int, int]] = []
        for fi in selected:
            face = self.faces[fi]
            n = len(face)
            for k in range(n):
                u, v = face[k], face[(k + 1) % n]
                if (v, u) not in directed_edges:
                    boundary_edges.append((u, v))

        # Duplicate every vertex touched by selected faces
        vert_map: Dict[int, int] = {}
        new_verts = list(self.vertices)
        offset = avg_normal * float(distance)

        for fi in selected:
            for v in self.faces[fi]:
                if v not in vert_map:
                    vert_map[v] = len(new_verts)
                    new_verts.append((self.vertices[v] + offset).astype(np.float32))

        self.vertices = np.asarray(new_verts, dtype=np.float32)

        # Replace selected cap faces with shifted cap faces
        for fi in selected:
            self.faces[fi] = [vert_map[v] for v in self.faces[fi]]

        # Stitch side quads along the boundary edges preserving outward winding
        for u, v in boundary_edges:
            u2 = vert_map[u]
            v2 = vert_map[v]
            self.faces.append([u, v, v2, u2])

        self.touch()
        return selected

    def inset_faces(self, face_indices: Sequence[int], factor: float = 0.25) -> List[int]:
        """Inset each selected face individually toward its centroid.

        Args:
            face_indices: Faces to inset.
            factor: Inset ratio in (0.01, 0.95).

        Returns:
            Indices of the new inner cap faces.
        """
        selected = sorted({i for i in face_indices if 0 <= i < len(self.faces)})
        if not selected:
            return []

        factor = float(np.clip(factor, 0.02, 0.95))
        new_verts = list(self.vertices)
        new_cap_indices: List[int] = []

        for fi in selected:
            face = list(self.faces[fi])
            center = self.face_center(fi)
            inner_indices: List[int] = []
            for v in face:
                p = self.vertices[v]
                inset_p = p + (center - p) * factor
                inner_indices.append(len(new_verts))
                new_verts.append(inset_p.astype(np.float32))

            # Replace original face with inner cap
            self.faces[fi] = inner_indices
            new_cap_indices.append(fi)

            # Add surrounding border quads
            n = len(face)
            for k in range(n):
                u0 = face[k]
                u1 = face[(k + 1) % n]
                i0 = inner_indices[k]
                i1 = inner_indices[(k + 1) % n]
                self.faces.append([u0, u1, i1, i0])

        self.vertices = np.asarray(new_verts, dtype=np.float32)
        self.touch()
        return new_cap_indices

    def subdivide_simple(self) -> None:
        """Subdivide every polygon into quads by adding face centers and edge midpoints (no smoothing)."""
        if not self.faces:
            return

        new_verts = list(self.vertices)
        edge_mid: Dict[Edge, int] = {}
        new_faces: List[List[int]] = []

        for face in self.faces:
            n = len(face)
            center_pos = self.vertices[face].mean(axis=0).astype(np.float32)
            center_idx = len(new_verts)
            new_verts.append(center_pos)

            mid_indices: List[int] = []
            for k in range(n):
                u = face[k]
                v = face[(k + 1) % n]
                e = canonical_edge(u, v)
                if e not in edge_mid:
                    mid_pos = ((self.vertices[u] + self.vertices[v]) * 0.5).astype(np.float32)
                    edge_mid[e] = len(new_verts)
                    new_verts.append(mid_pos)
                mid_indices.append(edge_mid[e])

            for k in range(n):
                prev_mid = mid_indices[(k - 1) % n]
                curr_v = face[k]
                curr_mid = mid_indices[k]
                new_faces.append([prev_mid, curr_v, curr_mid, center_idx])

        self.vertices = np.asarray(new_verts, dtype=np.float32)
        self.faces = new_faces
        self.touch()

    def subdivide_catmull_clark(self) -> None:
        """Apply one iteration of Catmull-Clark subdivision surface smoothing."""
        if not self.faces:
            return

        verts = self.vertices
        num_v = len(verts)

        # 1. Face points: average of each face's vertices
        face_points = np.array(
            [verts[f].mean(axis=0) for f in self.faces], dtype=np.float32
        )

        # Build edge -> adjacent faces and vertex -> adjacent faces / edges
        edge_faces: Dict[Edge, List[int]] = defaultdict(list)
        vert_faces: Dict[int, List[int]] = defaultdict(list)
        vert_edges: Dict[int, Set[Edge]] = defaultdict(set)

        for fi, face in enumerate(self.faces):
            n = len(face)
            for k in range(n):
                u = face[k]
                v = face[(k + 1) % n]
                e = canonical_edge(u, v)
                edge_faces[e].append(fi)
                vert_faces[u].append(fi)
                vert_edges[u].add(e)
                vert_edges[v].add(e)

        # 2. Edge points
        edge_point_pos: Dict[Edge, np.ndarray] = {}
        edge_midpoints: Dict[Edge, np.ndarray] = {}
        for e, adj_f in edge_faces.items():
            mid = (verts[e[0]] + verts[e[1]]) * 0.5
            edge_midpoints[e] = mid
            if len(adj_f) == 2:
                ep = (verts[e[0]] + verts[e[1]] + face_points[adj_f[0]] + face_points[adj_f[1]]) * 0.25
            else:
                # Boundary or non-manifold edge
                ep = mid
            edge_point_pos[e] = ep.astype(np.float32)

        # 3. Updated original vertex positions
        updated_verts = np.zeros_like(verts, dtype=np.float32)
        for vi in range(num_v):
            adj_f = vert_faces.get(vi, [])
            adj_e = vert_edges.get(vi, set())
            boundary_edges = [e for e in adj_e if len(edge_faces[e]) == 1]

            if not adj_f:
                updated_verts[vi] = verts[vi]
            elif boundary_edges:
                # Crease / boundary rule
                avg_b = np.mean([edge_midpoints[e] for e in boundary_edges], axis=0)
                updated_verts[vi] = (0.75 * verts[vi] + 0.25 * avg_b).astype(np.float32)
            else:
                n = float(len(adj_f))
                f_avg = np.mean([face_points[fi] for fi in adj_f], axis=0)
                r_avg = np.mean([edge_midpoints[e] for e in adj_e], axis=0)
                updated_verts[vi] = ((f_avg + 2.0 * r_avg + (n - 3.0) * verts[vi]) / n).astype(
                    np.float32
                )

        # Assemble new vertex array: [updated original verts | face points | edge points]
        new_verts: List[np.ndarray] = list(updated_verts)
        face_pt_idx: List[int] = []
        for fp in face_points:
            face_pt_idx.append(len(new_verts))
            new_verts.append(fp)

        edge_pt_idx: Dict[Edge, int] = {}
        for e, ep in edge_point_pos.items():
            edge_pt_idx[e] = len(new_verts)
            new_verts.append(ep)

        # 4. Build new quad faces
        new_faces: List[List[int]] = []
        for fi, face in enumerate(self.faces):
            n = len(face)
            fp_i = face_pt_idx[fi]
            for k in range(n):
                v_prev = face[(k - 1) % n]
                v_curr = face[k]
                v_next = face[(k + 1) % n]
                e_prev = edge_pt_idx[canonical_edge(v_prev, v_curr)]
                e_next = edge_pt_idx[canonical_edge(v_curr, v_next)]
                new_faces.append([e_prev, v_curr, e_next, fp_i])

        self.vertices = np.asarray(new_verts, dtype=np.float32)
        self.faces = new_faces
        self.touch()

    def bevel(
        self,
        factor: float = 0.22,
        edge_indices: Optional[Iterable[Edge]] = None,
    ) -> Set[Edge]:
        """Chamfer-bevel selected edges (or all manifold edges if edge_indices is None).

        Uses a sector-based vertex-fan split:
        - Vertices touching 0 selected edges remain untouched.
        - Vertices touching 1 selected edge cleanly split the two adjacent faces and cap the end.
        - Vertices touching 2 selected edges form a clean mitered quad strip (e.g. edge loop bevel).
        - Vertices touching 3+ selected edges form an outward-facing corner polygon.

        Returns the set of newly created bevel boundary edges.
        """
        if not self.faces or len(self.vertices) == 0:
            return set()

        factor = float(np.clip(factor, 0.04, 0.45))
        orig_verts = self.vertices
        orig_faces = [list(f) for f in self.faces]

        # Build half-edge -> face map and vertex -> incident faces
        directed_to_face: Dict[Tuple[int, int], int] = {}
        vert_incident_faces: Dict[int, List[int]] = defaultdict(list)
        all_manifold_edges: Set[Edge] = set()

        for fi, face in enumerate(orig_faces):
            n = len(face)
            for k in range(n):
                u, v = face[k], face[(k + 1) % n]
                directed_to_face[(u, v)] = fi
                vert_incident_faces[u].append(fi)

        for (u, v), f_a in directed_to_face.items():
            f_b = directed_to_face.get((v, u))
            if f_b is not None and f_b != f_a:
                all_manifold_edges.add(canonical_edge(u, v))

        if edge_indices is None:
            sel_edges = set(all_manifold_edges)
        else:
            sel_edges = {canonical_edge(u, v) for (u, v) in edge_indices} & all_manifold_edges

        if not sel_edges:
            return set()

        face_centers = [orig_verts[f].mean(axis=0).astype(np.float32) for f in orig_faces]

        # New vertex storage starts with copies of original vertices (unused ones pruned at end)
        new_verts: List[np.ndarray] = [v.copy() for v in orig_verts]

        # Replacement vertex sequence for each (face_idx, orig_vert_idx) corner
        # Usually a single [new_v], or [v_a, v_b] for cap faces at k==1 endpoints
        face_corner_repl: Dict[Tuple[int, int], List[int]] = {}
        # Specific vertex used by bevel quad for face fi at orig vertex vi
        bevel_corner_vert: Dict[Tuple[int, int], int] = {}
        extra_faces: List[List[int]] = []

        # Helper to find (prev_v, next_v) of vi inside face fi
        def face_neighbors(fi: int, vi: int) -> Tuple[int, int]:
            f = orig_faces[fi]
            idx = f.index(vi)
            return f[(idx - 1) % len(f)], f[(idx + 1) % len(f)]

        # Order incident faces around each vertex that touches at least one selected edge
        for vi, inc_faces in vert_incident_faces.items():
            # Check how many selected edges touch vi
            inc_sel = [
                e for e in sel_edges if e[0] == vi or e[1] == vi
            ]
            if not inc_sel:
                continue

            # Walk cyclic fan in CCW (outward) order: curr_f -> twin(prev_v -> vi) = (vi -> prev_v)
            in_nbr: Dict[int, int] = {}
            out_nbr: Dict[int, int] = {}
            for fi in inc_faces:
                pv, nv = face_neighbors(fi, vi)
                in_nbr[fi] = pv
                out_nbr[fi] = nv

            # Find a starting face whose incoming edge (or boundary) starts a fan
            fan: List[int] = []
            visited_f: Set[int] = set()
            # Prefer starting at a boundary face (no twin for outgoing edge) if non-closed fan
            start_f = inc_faces[0]
            for fi in inc_faces:
                if directed_to_face.get((out_nbr[fi], vi)) is None:
                    start_f = fi
                    break

            curr: Optional[int] = start_f
            while curr is not None and curr not in visited_f:
                visited_f.add(curr)
                fan.append(curr)
                curr = directed_to_face.get((vi, in_nbr[curr]))

            is_closed_fan = curr == start_f and len(fan) == len(inc_faces)
            if not is_closed_fan or len(fan) < 2:
                # Fallback for open-boundary vertices: slide each incident face corner toward its face center
                for fi in inc_faces:
                    pos = (orig_verts[vi] + (face_centers[fi] - orig_verts[vi]) * factor).astype(np.float32)
                    n_idx = len(new_verts)
                    new_verts.append(pos)
                    face_corner_repl[(fi, vi)] = [n_idx]
                    bevel_corner_vert[(fi, vi)] = n_idx
                continue

            # Identify cut positions in the closed CCW fan.
            # Between fan[i] and fan[(i + 1) % m], the shared edge is (vi, in_nbr[fan[i]]).
            m = len(fan)
            cut_indices = [
                i for i in range(m)
                if canonical_edge(vi, in_nbr[fan[i]]) in sel_edges
            ]
            k = len(cut_indices)
            if k == 0:
                continue

            # Rotate fan so it starts right after the last cut (i.e. fan[0] has cut on its outgoing edge)
            start_pos = (cut_indices[-1] + 1) % m
            fan = fan[start_pos:] + fan[:start_pos]

            # Split fan into k contiguous sectors bounded by selected edges
            sectors: List[List[int]] = []
            cur_sector: List[int] = []
            for i, fi in enumerate(fan):
                cur_sector.append(fi)
                if canonical_edge(vi, in_nbr[fi]) in sel_edges:
                    sectors.append(cur_sector)
                    cur_sector = []
            if cur_sector:
                sectors.append(cur_sector)

            p_v = orig_verts[vi]

            if k == 1:
                # Single selected edge ends at vi; sectors == [fan]
                # fan[0] has outgoing selected edge; fan[-1] has incoming selected edge.
                f_first = fan[0]
                f_last = fan[-1]
                # Edge between fan[0] and fan[1] is (vi, in_nbr[fan[0]])
                u_first = in_nbr[f_first]
                # Edge between fan[-2] and fan[-1] is (vi, out_nbr[fan[-1]])
                u_last = out_nbr[f_last]

                pos_first = (p_v + (orig_verts[u_first] - p_v) * (factor * 0.5)).astype(np.float32)
                pos_last = (p_v + (orig_verts[u_last] - p_v) * (factor * 0.5)).astype(np.float32)

                v_first = len(new_verts)
                new_verts.append(pos_first)
                v_last = len(new_verts)
                new_verts.append(pos_last)

                face_corner_repl[(f_first, vi)] = [v_first]
                bevel_corner_vert[(f_first, vi)] = v_first
                face_corner_repl[(f_last, vi)] = [v_last]
                bevel_corner_vert[(f_last, vi)] = v_last

                if m == 3:
                    # Clean 3-valence corner (e.g. cube): middle face absorbs the chamfer edge [v_last, v_first]
                    f_mid = fan[1]
                    face_corner_repl[(f_mid, vi)] = [v_last, v_first]
                elif m >= 4:
                    # 4+-valence vertex: keep vi for interior faces, split adjacent edges, cap with triangle
                    face_corner_repl[(fan[1], vi)] = [vi, v_first]
                    face_corner_repl[(fan[-2], vi)] = [v_last, vi]
                    extra_faces.append([v_last, v_first, vi])
            else:
                # k >= 2: each sector gets one new vertex
                sector_verts: List[int] = []
                for sec in sectors:
                    if len(sec) == 1:
                        fi = sec[0]
                        pos = (p_v + (face_centers[fi] - p_v) * factor).astype(np.float32)
                    else:
                        # Slide along the average of internal unselected edges inside this sector
                        dirs = [
                            orig_verts[in_nbr[sec[j]]] - p_v
                            for j in range(len(sec) - 1)
                        ]
                        avg_dir = np.mean(dirs, axis=0)
                        pos = (p_v + avg_dir * (factor * 0.5)).astype(np.float32)
                    v_new = len(new_verts)
                    new_verts.append(pos)
                    sector_verts.append(v_new)
                    for fi in sec:
                        face_corner_repl[(fi, vi)] = [v_new]
                        bevel_corner_vert[(fi, vi)] = v_new

                if k >= 3:
                    # Outward CCW corner polygon
                    extra_faces.append(list(sector_verts))

        # Rebuild updated original faces
        new_faces: List[List[int]] = []
        for fi, face in enumerate(orig_faces):
            rebuilt: List[int] = []
            for vi in face:
                repl = face_corner_repl.get((fi, vi))
                if repl is not None:
                    rebuilt.extend(repl)
                else:
                    rebuilt.append(vi)
            if len(rebuilt) >= 3:
                new_faces.append(rebuilt)

        # Create bevel quads along every selected edge
        created_edges: Set[Edge] = set()
        processed: Set[Edge] = set()
        for (u, v), f_a in directed_to_face.items():
            e = canonical_edge(u, v)
            if e not in sel_edges or e in processed:
                continue
            processed.add(e)
            f_b = directed_to_face.get((v, u))
            if f_b is None or f_b == f_a:
                continue
            ca_v = bevel_corner_vert[(f_a, v)]
            ca_u = bevel_corner_vert[(f_a, u)]
            cb_u = bevel_corner_vert[(f_b, u)]
            cb_v = bevel_corner_vert[(f_b, v)]
            new_faces.append([ca_v, ca_u, cb_u, cb_v])
            created_edges.add(canonical_edge(ca_v, ca_u))
            created_edges.add(canonical_edge(cb_u, cb_v))

        new_faces.extend(extra_faces)

        self.vertices = np.asarray(new_verts, dtype=np.float32)
        self.faces = new_faces
        # Prune unreferenced original vertices while tracking returned edges
        used = sorted({v for f in self.faces for v in f})
        remap = {old: new for new, old in enumerate(used)}
        self.vertices = self.vertices[used].astype(np.float32)
        self.faces = [[remap[v] for v in f] for f in self.faces]
        remapped_edges = {
            canonical_edge(remap[u], remap[v])
            for (u, v) in created_edges
            if u in remap and v in remap and remap[u] != remap[v]
        }
        self.touch()
        return remapped_edges

    def poke_faces(
        self, face_indices: Optional[Sequence[int]] = None, offset: float = 0.0
    ) -> List[int]:
        """Insert a center vertex into each selected face (optionally offset along its normal) and fan triangles."""
        if not self.faces:
            return []
        targets = (
            set(range(len(self.faces)))
            if face_indices is None
            else {i for i in face_indices if 0 <= i < len(self.faces)}
        )
        if not targets:
            return []

        new_verts = list(self.vertices)
        new_faces: List[List[int]] = []
        created_center_verts: List[int] = []

        for fi, face in enumerate(self.faces):
            if fi not in targets:
                new_faces.append(list(face))
                continue
            center = self.face_center(fi)
            if abs(offset) > 1e-6:
                center = (center + self.face_normal(fi) * float(offset)).astype(np.float32)
            c_idx = len(new_verts)
            new_verts.append(center)
            created_center_verts.append(c_idx)
            n = len(face)
            for k in range(n):
                new_faces.append([face[k], face[(k + 1) % n], c_idx])

        self.vertices = np.asarray(new_verts, dtype=np.float32)
        self.faces = new_faces
        self.touch()
        return created_center_verts

    def loop_cut(
        self, seed_edge: Optional[Edge] = None, cuts: int = 1
    ) -> Set[Edge]:
        """Slice a quad edge ring with `cuts` parallel edge loops. Returns the newly created cut edges."""
        cuts = max(1, min(16, int(cuts)))
        if not self.faces:
            return set()

        # Build undirected edge -> incident quad faces
        edge_to_quads: Dict[Edge, List[int]] = defaultdict(list)
        for fi, face in enumerate(self.faces):
            if len(face) != 4:
                continue
            for k in range(4):
                edge_to_quads[canonical_edge(face[k], face[(k + 1) % 4])].append(fi)

        if not edge_to_quads:
            return set()

        # Choose default seed edge if none supplied: longest quad edge
        if seed_edge is None or canonical_edge(*seed_edge) not in edge_to_quads:
            seed_edge = max(
                edge_to_quads.keys(),
                key=lambda e: float(np.linalg.norm(self.vertices[e[1]] - self.vertices[e[0]])),
            )
        else:
            seed_edge = canonical_edge(*seed_edge)

        # Helper to find the opposite edge inside a quad
        def quad_opposite_edge(fi: int, e: Edge) -> Tuple[Edge, Tuple[int, int], Tuple[int, int]]:
            face = self.faces[fi]
            for k in range(4):
                a, b = face[k], face[(k + 1) % 4]
                if canonical_edge(a, b) == e:
                    # Opposite edge keeping parallel direction: d -> c corresponds to a -> b
                    c, d = face[(k + 2) % 4], face[(k + 3) % 4]
                    return canonical_edge(d, c), (a, b), (d, c)
            raise ValueError("Edge not in quad")

        # Walk the edge ring in both directions from seed_edge
        ring_edges: Set[Edge] = {seed_edge}
        ring_quads: Set[int] = set()
        queue = deque([seed_edge])
        while queue:
            curr_e = queue.popleft()
            for fi in edge_to_quads.get(curr_e, []):
                if fi in ring_quads:
                    continue
                ring_quads.add(fi)
                opp_e, _, _ = quad_opposite_edge(fi, curr_e)
                if opp_e not in ring_edges:
                    ring_edges.add(opp_e)
                    queue.append(opp_e)

        if not ring_quads:
            return set()

        # Split every ring edge into (cuts + 1) segments using canonical (low -> high) order
        new_verts = list(self.vertices)
        edge_split_verts: Dict[Edge, List[int]] = {}
        for e in ring_edges:
            u, v = e
            pu, pv = self.vertices[u], self.vertices[v]
            chain = [u]
            for s in range(1, cuts + 1):
                t = s / float(cuts + 1)
                idx = len(new_verts)
                new_verts.append((pu * (1.0 - t) + pv * t).astype(np.float32))
                chain.append(idx)
            chain.append(v)
            edge_split_verts[e] = chain

        def oriented_chain(a: int, b: int) -> List[int]:
            e = canonical_edge(a, b)
            chain = edge_split_verts[e]
            return chain if chain[0] == a else list(reversed(chain))

        # Rebuild faces: replace each ring quad with (cuts + 1) sub-quads
        new_faces: List[List[int]] = []
        created_cut_edges: Set[Edge] = set()

        for fi, face in enumerate(self.faces):
            if fi not in ring_quads:
                # If a non-ring face shares a split edge, insert the split vertices along that edge
                updated_face: List[int] = []
                n = len(face)
                for k in range(n):
                    a, b = face[k], face[(k + 1) % n]
                    e = canonical_edge(a, b)
                    if e in edge_split_verts:
                        chain = oriented_chain(a, b)
                        updated_face.extend(chain[:-1])
                    else:
                        updated_face.append(a)
                new_faces.append(updated_face)
                continue

            # Find which opposing edge pair of this quad is in ring_edges
            e01 = canonical_edge(face[0], face[1])
            if e01 in ring_edges:
                c_bot = oriented_chain(face[0], face[1])
                c_top = oriented_chain(face[3], face[2])
            else:
                c_bot = oriented_chain(face[1], face[2])
                c_top = oriented_chain(face[0], face[3])

            for s in range(cuts + 1):
                new_faces.append([c_bot[s], c_bot[s + 1], c_top[s + 1], c_top[s]])
                if s < cuts:
                    created_cut_edges.add(canonical_edge(c_bot[s + 1], c_top[s + 1]))

        self.vertices = np.asarray(new_verts, dtype=np.float32)
        self.faces = new_faces
        self.touch()
        return created_cut_edges

    def bridge_faces(self, face_a_idx: int, face_b_idx: int) -> bool:
        """Delete two faces with the same vertex count and bridge their boundary loops with quads."""
        if face_a_idx == face_b_idx:
            return False
        if not (0 <= face_a_idx < len(self.faces) and 0 <= face_b_idx < len(self.faces)):
            return False
        fa = list(self.faces[face_a_idx])
        fb = list(self.faces[face_b_idx])
        n = len(fa)
        if n < 3 or len(fb) != n:
            return False

        # Reverse fb so its loop traverses in the same cyclic direction as fa
        fb_rev = list(reversed(fb))

        # Pick cyclic rotation of fb_rev minimizing total squared bridge edge length (no twist)
        best_shift = 0
        best_cost = float("inf")
        for shift in range(n):
            cost = 0.0
            for k in range(n):
                diff = self.vertices[fa[k]] - self.vertices[fb_rev[(k + shift) % n]]
                cost += float(np.dot(diff, diff))
            if cost < best_cost:
                best_cost = cost
                best_shift = shift

        fb_aligned = [fb_rev[(k + best_shift) % n] for k in range(n)]

        # Remove the two bridged cap faces
        remove = {face_a_idx, face_b_idx}
        self.faces = [f for i, f in enumerate(self.faces) if i not in remove]

        # Stitch side quads with outward normals
        for k in range(n):
            u0 = fa[k]
            u1 = fa[(k + 1) % n]
            v0 = fb_aligned[k]
            v1 = fb_aligned[(k + 1) % n]
            self.faces.append([u0, u1, v1, v0])

        self.touch()
        return True

    def solidify(self, thickness: float = 0.16) -> None:
        """Give shell thickness to a mesh by offsetting a flipped inner copy and bridging open boundaries."""
        if not self.faces or len(self.vertices) == 0:
            return
        if abs(thickness) < 1e-6:
            return

        orig_count = len(self.vertices)
        v_normals = self.compute_vertex_normals()
        offset_verts = (self.vertices - v_normals * float(thickness)).astype(np.float32)

        # Find directed open boundary edges (u, v) where (v, u) does not exist
        directed: Set[Tuple[int, int]] = set()
        for face in self.faces:
            n = len(face)
            for k in range(n):
                directed.add((face[k], face[(k + 1) % n]))

        boundary_edges = [(u, v) for (u, v) in directed if (v, u) not in directed]

        orig_faces = [list(f) for f in self.faces]
        inner_faces = [[v + orig_count for v in reversed(f)] for f in orig_faces]
        rim_faces = [[v, u, u + orig_count, v + orig_count] for (u, v) in boundary_edges]

        self.vertices = np.vstack([self.vertices, offset_verts]).astype(np.float32)
        self.faces = orig_faces + inner_faces + rim_faces
        self.touch()

    def mirror(self, axis: int = 0, weld_threshold: float = 1e-3) -> None:
        """Mirror mesh across X (0), Y (1), or Z (2) plane, flip mirrored winding, and weld seam."""
        if not self.faces or len(self.vertices) == 0:
            return
        axis = int(axis) % 3
        orig_count = len(self.vertices)

        mirrored = self.vertices.copy()
        mirrored[:, axis] = -mirrored[:, axis]

        mirrored_faces = [[v + orig_count for v in reversed(f)] for f in self.faces]

        self.vertices = np.vstack([self.vertices, mirrored]).astype(np.float32)
        self.faces = [list(f) for f in self.faces] + mirrored_faces
        if weld_threshold > 0.0:
            self.merge_by_distance(weld_threshold)
        else:
            self.touch()

    def radial_array(self, count: int = 6, radius: float = 0.0) -> None:
        """Duplicate the mesh radially around the Y axis `count` times."""
        count = max(2, min(48, int(count)))
        if not self.faces or len(self.vertices) == 0:
            return

        base_verts = self.vertices.copy()
        if abs(radius) > 1e-6:
            base_verts[:, 0] += float(radius)
        base_faces = [list(f) for f in self.faces]
        n_v = len(base_verts)

        all_verts: List[np.ndarray] = []
        all_faces: List[List[int]] = []

        for i in range(count):
            ang = 2.0 * math.pi * i / count
            c, s = math.cos(ang), math.sin(ang)
            rot_y = np.array(
                [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32
            )
            all_verts.append((base_verts @ rot_y.T).astype(np.float32))
            off = i * n_v
            for f in base_faces:
                all_faces.append([vi + off for vi in f])

        self.vertices = np.vstack(all_verts).astype(np.float32)
        self.faces = all_faces
        self.touch()

    def smooth_vertices(
        self,
        vert_indices: Optional[Iterable[int]] = None,
        factor: float = 0.5,
        iterations: int = 2,
    ) -> None:
        """Apply Laplacian smoothing to selected vertices (or the entire mesh)."""
        if len(self.vertices) == 0 or not self.faces:
            return
        factor = float(np.clip(factor, 0.05, 1.0))
        iterations = max(1, min(20, int(iterations)))

        neighbors: Dict[int, Set[int]] = defaultdict(set)
        for face in self.faces:
            n = len(face)
            for k in range(n):
                u, v = face[k], face[(k + 1) % n]
                neighbors[u].add(v)
                neighbors[v].add(u)

        targets = (
            range(len(self.vertices))
            if vert_indices is None
            else [vi for vi in set(vert_indices) if 0 <= vi < len(self.vertices)]
        )

        verts = self.vertices.copy()
        for _ in range(iterations):
            updated = verts.copy()
            for vi in targets:
                nbrs = neighbors.get(vi)
                if not nbrs:
                    continue
                avg = verts[list(nbrs)].mean(axis=0)
                updated[vi] = verts[vi] + (avg - verts[vi]) * factor
            verts = updated

        self.vertices = verts.astype(np.float32)
        self.touch()

    def inflate_vertices(
        self,
        distance: float = 0.15,
        vert_indices: Optional[Iterable[int]] = None,
    ) -> None:
        """Displace target vertices outward (or inward if negative) along their smooth vertex normals."""
        if len(self.vertices) == 0 or not self.faces:
            return
        v_normals = self.compute_vertex_normals()
        if vert_indices is None:
            self.vertices = (self.vertices + v_normals * float(distance)).astype(np.float32)
        else:
            idx = [vi for vi in set(vert_indices) if 0 <= vi < len(self.vertices)]
            if not idx:
                return
            verts = self.vertices.copy()
            verts[idx] += v_normals[idx] * float(distance)
            self.vertices = verts.astype(np.float32)
        self.touch()

    def twist(self, angle_deg: float = 90.0, axis: int = 1) -> None:
        """Twist the mesh progressively along `axis` (default Y) across its bounding height."""
        if len(self.vertices) == 0:
            return
        axis = int(axis) % 3
        a1, a2 = [i for i in range(3) if i != axis]
        bmin, bmax = self.compute_bounds()
        span = float(bmax[axis] - bmin[axis])
        if span < 1e-6:
            return
        center = (bmin + bmax) * 0.5
        angle_rad = math.radians(angle_deg)

        verts = self.vertices.copy()
        t = (verts[:, axis] - bmin[axis]) / span - 0.5
        angles = t * angle_rad
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        dx = verts[:, a1] - center[a1]
        dz = verts[:, a2] - center[a2]
        verts[:, a1] = center[a1] + dx * cos_a - dz * sin_a
        verts[:, a2] = center[a2] + dx * sin_a + dz * cos_a
        self.vertices = verts.astype(np.float32)
        self.touch()

    def taper(self, factor: float = 0.6, axis: int = 1) -> None:
        """Taper the cross-section linearly from bottom to top along `axis` (default Y)."""
        if len(self.vertices) == 0:
            return
        axis = int(axis) % 3
        a1, a2 = [i for i in range(3) if i != axis]
        bmin, bmax = self.compute_bounds()
        span = float(bmax[axis] - bmin[axis])
        if span < 1e-6:
            return
        center = (bmin + bmax) * 0.5

        verts = self.vertices.copy()
        t = (verts[:, axis] - bmin[axis]) / span  # 0 at bottom, 1 at top
        scale = np.clip(1.0 + (t - 0.5) * float(factor), 0.05, 5.0)
        verts[:, a1] = center[a1] + (verts[:, a1] - center[a1]) * scale
        verts[:, a2] = center[a2] + (verts[:, a2] - center[a2]) * scale
        self.vertices = verts.astype(np.float32)
        self.touch()

    def merge_by_distance(self, threshold: float = 0.01) -> int:
        """Weld vertices closer than `threshold` and clean degenerate faces. Returns merged count."""
        if len(self.vertices) == 0:
            return 0
        thr = max(1e-6, float(threshold))
        inv_thr = 1.0 / thr

        # Spatial hash clustering for fast welding
        buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        remap = list(range(len(self.vertices)))
        new_verts: List[np.ndarray] = []

        for vi, pt in enumerate(self.vertices):
            gx = int(math.floor(float(pt[0]) * inv_thr))
            gy = int(math.floor(float(pt[1]) * inv_thr))
            gz = int(math.floor(float(pt[2]) * inv_thr))
            found = -1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for cand in buckets.get((gx + dx, gy + dy, gz + dz), ()):
                            if float(np.linalg.norm(pt - new_verts[cand])) <= thr:
                                found = cand
                                break
                        if found >= 0:
                            break
                    if found >= 0:
                        break
                if found >= 0:
                    break

            if found >= 0:
                remap[vi] = found
            else:
                new_idx = len(new_verts)
                new_verts.append(pt.copy())
                buckets[(gx, gy, gz)].append(new_idx)
                remap[vi] = new_idx

        merged_count = len(self.vertices) - len(new_verts)

        cleaned_faces: List[List[int]] = []
        for face in self.faces:
            mapped = [remap[v] for v in face]
            # Collapse consecutive duplicates
            dedup: List[int] = []
            for v in mapped:
                if not dedup or dedup[-1] != v:
                    dedup.append(v)
            if len(dedup) > 1 and dedup[0] == dedup[-1]:
                dedup.pop()
            if len(dedup) >= 3 and len(set(dedup)) >= 3:
                cleaned_faces.append(dedup)

        self.vertices = np.asarray(new_verts, dtype=np.float32) if new_verts else np.zeros((0, 3), dtype=np.float32)
        self.faces = cleaned_faces
        self._prune_unused_vertices()
        self.touch()
        return merged_count

    def recalculate_normals(self) -> None:
        """Consistently orient all face normals outward per connected component."""
        if not self.faces:
            return

        # Build edge -> [(face_idx, u, v)]
        edge_uses: Dict[Edge, List[Tuple[int, int, int]]] = defaultdict(list)
        for fi, face in enumerate(self.faces):
            n = len(face)
            for k in range(n):
                u, v = face[k], face[(k + 1) % n]
                edge_uses[canonical_edge(u, v)].append((fi, u, v))

        faces = [list(f) for f in self.faces]
        visited: Set[int] = set()

        for start_f in range(len(faces)):
            if start_f in visited:
                continue
            comp: List[int] = []
            queue = deque([start_f])
            visited.add(start_f)
            while queue:
                fi = queue.popleft()
                comp.append(fi)
                f = faces[fi]
                n = len(f)
                for k in range(n):
                    u, v = f[k], f[(k + 1) % n]
                    for other_fi, ou, ov in edge_uses[canonical_edge(u, v)]:
                        if other_fi in visited:
                            continue
                        # Adjacent faces should traverse shared edge in opposite directions
                        if (ou, ov) == (u, v):
                            faces[other_fi] = list(reversed(faces[other_fi]))
                        visited.add(other_fi)
                        queue.append(other_fi)

            # Check signed volume of the component (positive -> outward CCW)
            signed_vol6 = 0.0
            for fi in comp:
                f = faces[fi]
                v0 = self.vertices[f[0]]
                for k in range(1, len(f) - 1):
                    v1 = self.vertices[f[k]]
                    v2 = self.vertices[f[k + 1]]
                    signed_vol6 += float(np.dot(v0, np.cross(v1, v2)))
            if signed_vol6 < 0.0:
                for fi in comp:
                    faces[fi] = list(reversed(faces[fi]))

        self.faces = faces
        self.touch()

    # ------------------------------------------------------------------
    # Topological Selection Helpers
    # ------------------------------------------------------------------

    def find_edge_loop(self, seed_edge: Edge) -> Set[Edge]:
        """Expand a single edge into its topological edge loop through 4-valence vertices."""
        seed = canonical_edge(*seed_edge)
        vert_edges: Dict[int, Set[Edge]] = defaultdict(set)
        edge_faces: Dict[Edge, Set[int]] = defaultdict(set)

        for fi, face in enumerate(self.faces):
            n = len(face)
            for k in range(n):
                e = canonical_edge(face[k], face[(k + 1) % n])
                vert_edges[e[0]].add(e)
                vert_edges[e[1]].add(e)
                edge_faces[e].add(fi)

        if seed not in edge_faces:
            return set()

        loop: Set[Edge] = {seed}

        def walk(start_edge: Edge, tip_vertex: int) -> None:
            curr_e = start_edge
            curr_v = tip_vertex
            while True:
                incident = vert_edges.get(curr_v, set())
                if len(incident) != 4:
                    break
                curr_adj_faces = edge_faces[curr_e]
                # Opposite edge at a 4-valence vertex shares NO faces with curr_e
                candidates = [
                    e for e in incident
                    if e != curr_e and edge_faces[e].isdisjoint(curr_adj_faces)
                ]
                if len(candidates) != 1:
                    break
                nxt = candidates[0]
                if nxt in loop:
                    break
                loop.add(nxt)
                curr_v = nxt[1] if nxt[0] == curr_v else nxt[0]
                curr_e = nxt

        walk(seed, seed[0])
        walk(seed, seed[1])
        return loop

    def grow_vertex_selection(self, selected: Set[int]) -> Set[int]:
        out = set(selected)
        for face in self.faces:
            if any(v in selected for v in face):
                out.update(face)
        return out

    def grow_edge_selection(self, selected: Set[Edge]) -> Set[Edge]:
        verts = {v for e in selected for v in e}
        out = set(selected)
        for face in self.faces:
            n = len(face)
            for k in range(n):
                u, v = face[k], face[(k + 1) % n]
                if u in verts or v in verts:
                    out.add(canonical_edge(u, v))
        return out

    def grow_face_selection(self, selected: Set[int]) -> Set[int]:
        verts: Set[int] = set()
        for fi in selected:
            if 0 <= fi < len(self.faces):
                verts.update(self.faces[fi])
        out = set(selected)
        for fi, face in enumerate(self.faces):
            if any(v in verts for v in face):
                out.add(fi)
        return out

    def to_dict(self) -> dict:
        return {
            "vertices": self.vertices.tolist(),
            "faces": [list(f) for f in self.faces],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mesh":
        return cls(data.get("vertices", []), data.get("faces", []))
