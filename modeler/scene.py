"""Scene graph, materials, selection state, and undo/redo history."""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .math3d import transform_points, trs_matrix
from .mesh import Edge, Mesh, canonical_edge
from .primitives import create_cube


class EditMode(str, Enum):
    OBJECT = "object"
    VERTEX = "vertex"
    EDGE = "edge"
    FACE = "face"


class TransformTool(str, Enum):
    SELECT = "select"
    MOVE = "move"
    ROTATE = "rotate"
    SCALE = "scale"


class ShadingMode(str, Enum):
    SOLID = "solid"
    SOLID_WIRE = "solid_wire"
    WIREFRAME = "wireframe"
    SMOOTH = "smooth"


@dataclass
class Material:
    name: str = "Default Clay"
    color: Tuple[float, float, float] = (0.58, 0.68, 0.82)
    roughness: float = 0.45
    metallic: float = 0.15
    wireframe_color: Tuple[float, float, float] = (0.12, 0.15, 0.20)

    def clone(self) -> "Material":
        return Material(
            name=self.name,
            color=tuple(self.color),
            roughness=float(self.roughness),
            metallic=float(self.metallic),
            wireframe_color=tuple(self.wireframe_color),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": list(self.color),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "wireframe_color": list(self.wireframe_color),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        return cls(
            name=data.get("name", "Material"),
            color=tuple(data.get("color", [0.6, 0.7, 0.85])),
            roughness=float(data.get("roughness", 0.45)),
            metallic=float(data.get("metallic", 0.15)),
            wireframe_color=tuple(data.get("wireframe_color", [0.12, 0.15, 0.2])),
        )


MATERIAL_PRESETS: Dict[str, Material] = {
    "Studio Clay": Material("Studio Clay", (0.72, 0.74, 0.78), 0.65, 0.05),
    "Cobalt Blue": Material("Cobalt Blue", (0.28, 0.56, 0.88), 0.35, 0.20),
    "Brushed Steel": Material("Brushed Steel", (0.76, 0.80, 0.85), 0.25, 0.80),
    "Polished Gold": Material("Polished Gold", (0.94, 0.76, 0.32), 0.20, 0.88),
    "Warm Copper": Material("Warm Copper", (0.85, 0.48, 0.34), 0.28, 0.78),
    "Emerald": Material("Emerald", (0.22, 0.72, 0.52), 0.30, 0.25),
    "Terracotta": Material("Terracotta", (0.82, 0.42, 0.32), 0.72, 0.05),
    "Obsidian": Material("Obsidian", (0.18, 0.20, 0.25), 0.18, 0.45),
}


@dataclass
class SceneObject:
    name: str
    mesh: Mesh
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    material: Material = field(default_factory=Material)
    visible: bool = True
    locked: bool = False
    obj_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def model_matrix(self) -> np.ndarray:
        return trs_matrix(self.position, self.rotation, self.scale)

    def world_vertices(self) -> np.ndarray:
        return transform_points(self.mesh.vertices, self.model_matrix())

    def world_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        wv = self.world_vertices()
        if len(wv) == 0:
            p = np.asarray(self.position, dtype=np.float32)
            return p, p.copy()
        return wv.min(axis=0).astype(np.float32), wv.max(axis=0).astype(np.float32)

    def clone(self, new_id: bool = True, new_name: Optional[str] = None) -> "SceneObject":
        return SceneObject(
            name=new_name or self.name,
            mesh=self.mesh.clone(),
            position=list(self.position),
            rotation=list(self.rotation),
            scale=list(self.scale),
            material=self.material.clone(),
            visible=self.visible,
            locked=self.locked,
            obj_id=uuid.uuid4().hex[:10] if new_id else self.obj_id,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.obj_id,
            "name": self.name,
            "position": [float(v) for v in self.position],
            "rotation": [float(v) for v in self.rotation],
            "scale": [float(v) for v in self.scale],
            "visible": bool(self.visible),
            "locked": bool(self.locked),
            "material": self.material.to_dict(),
            "mesh": self.mesh.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneObject":
        return cls(
            obj_id=data.get("id", uuid.uuid4().hex[:10]),
            name=data.get("name", "Object"),
            position=[float(v) for v in data.get("position", [0.0, 0.0, 0.0])],
            rotation=[float(v) for v in data.get("rotation", [0.0, 0.0, 0.0])],
            scale=[float(v) for v in data.get("scale", [1.0, 1.0, 1.0])],
            visible=bool(data.get("visible", True)),
            locked=bool(data.get("locked", False)),
            material=Material.from_dict(data.get("material", {})),
            mesh=Mesh.from_dict(data.get("mesh", {})),
        )


@dataclass
class _SceneSnapshot:
    label: str
    objects: List[SceneObject]
    selected_object_ids: List[str]
    active_object_id: Optional[str]
    selected_vertices: Set[int]
    selected_edges: Set[Edge]
    selected_faces: Set[int]


class Scene:
    """Manages scene objects, selection, edit mode, and undo/redo history."""

    MAX_HISTORY = 60

    def __init__(self) -> None:
        self.objects: List[SceneObject] = []
        self.selected_object_ids: List[str] = []
        self.active_object_id: Optional[str] = None

        self.edit_mode: EditMode = EditMode.OBJECT
        self.tool: TransformTool = TransformTool.MOVE
        self.shading: ShadingMode = ShadingMode.SOLID_WIRE
        self.show_grid: bool = True
        self.snap_enabled: bool = False
        self.snap_increment: float = 0.25

        # Sub-object selection for active object in Edit mode
        self.selected_vertices: Set[int] = set()
        self.selected_edges: Set[Edge] = set()
        self.selected_faces: Set[int] = set()

        self._undo_stack: List[_SceneSnapshot] = []
        self._redo_stack: List[_SceneSnapshot] = []

    def reset_default_scene(self) -> None:
        """Initialize a clean starter scene with a centered cube."""
        self.objects.clear()
        self.selected_object_ids.clear()
        self.active_object_id = None
        self.clear_sub_selection()
        self.edit_mode = EditMode.OBJECT
        self._undo_stack.clear()
        self._redo_stack.clear()

        cube = SceneObject(
            name="Cube",
            mesh=create_cube(1.6),
            position=[0.0, 0.8, 0.0],
            material=MATERIAL_PRESETS["Studio Clay"].clone(),
        )
        self.objects.append(cube)
        self.select_object(cube.obj_id)

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _capture_snapshot(self, label: str) -> _SceneSnapshot:
        return _SceneSnapshot(
            label=label,
            objects=[o.clone(new_id=False) for o in self.objects],
            selected_object_ids=list(self.selected_object_ids),
            active_object_id=self.active_object_id,
            selected_vertices=set(self.selected_vertices),
            selected_edges=set(self.selected_edges),
            selected_faces=set(self.selected_faces),
        )

    def _restore_snapshot(self, snap: _SceneSnapshot) -> None:
        self.objects = [o.clone(new_id=False) for o in snap.objects]
        self.selected_object_ids = list(snap.selected_object_ids)
        self.active_object_id = snap.active_object_id
        self.selected_vertices = set(snap.selected_vertices)
        self.selected_edges = set(snap.selected_edges)
        self.selected_faces = set(snap.selected_faces)
        if self.active_object() is None and self.edit_mode != EditMode.OBJECT:
            self.edit_mode = EditMode.OBJECT

    def push_undo(self, label: str = "Edit") -> None:
        """Record the current scene state before performing a mutating operation."""
        self._undo_stack.append(self._capture_snapshot(label))
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[str]:
        if not self._undo_stack:
            return None
        snap = self._undo_stack.pop()
        self._redo_stack.append(self._capture_snapshot(snap.label))
        self._restore_snapshot(snap)
        return snap.label

    def redo(self) -> Optional[str]:
        if not self._redo_stack:
            return None
        snap = self._redo_stack.pop()
        self._undo_stack.append(self._capture_snapshot(snap.label))
        self._restore_snapshot(snap)
        return snap.label

    # ------------------------------------------------------------------
    # Object Management & Selection
    # ------------------------------------------------------------------

    def unique_name(self, base: str) -> str:
        existing = {o.name for o in self.objects}
        if base not in existing:
            return base
        idx = 1
        while f"{base}.{idx:03d}" in existing:
            idx += 1
        return f"{base}.{idx:03d}"

    def add_object(self, obj: SceneObject, record_undo: bool = True) -> SceneObject:
        if record_undo:
            self.push_undo(f"Add {obj.name}")
        obj.name = self.unique_name(obj.name)
        self.objects.append(obj)
        self.select_object(obj.obj_id)
        return obj

    def get_object(self, obj_id: Optional[str]) -> Optional[SceneObject]:
        if obj_id is None:
            return None
        for obj in self.objects:
            if obj.obj_id == obj_id:
                return obj
        return None

    def active_object(self) -> Optional[SceneObject]:
        obj = self.get_object(self.active_object_id)
        if obj is not None:
            return obj
        if self.selected_object_ids:
            return self.get_object(self.selected_object_ids[-1])
        return None

    def selected_objects(self) -> List[SceneObject]:
        id_set = set(self.selected_object_ids)
        return [o for o in self.objects if o.obj_id in id_set]

    def select_object(self, obj_id: Optional[str], additive: bool = False) -> None:
        if obj_id is None:
            if not additive:
                self.selected_object_ids.clear()
                self.active_object_id = None
                self.clear_sub_selection()
            return

        if not additive:
            self.selected_object_ids = [obj_id]
            self.active_object_id = obj_id
        else:
            if obj_id in self.selected_object_ids:
                self.selected_object_ids.remove(obj_id)
                if self.active_object_id == obj_id:
                    self.active_object_id = (
                        self.selected_object_ids[-1] if self.selected_object_ids else None
                    )
            else:
                self.selected_object_ids.append(obj_id)
                self.active_object_id = obj_id
        self.clear_sub_selection()

    def clear_sub_selection(self) -> None:
        self.selected_vertices.clear()
        self.selected_edges.clear()
        self.selected_faces.clear()

    def set_edit_mode(self, mode: EditMode) -> bool:
        if mode != EditMode.OBJECT and self.active_object() is None:
            return False
        if self.edit_mode != mode:
            self.edit_mode = mode
            self.clear_sub_selection()
        return True

    def delete_selected_objects(self) -> int:
        if not self.selected_object_ids:
            return 0
        self.push_undo("Delete Objects")
        del_set = set(self.selected_object_ids)
        before = len(self.objects)
        self.objects = [o for o in self.objects if o.obj_id not in del_set]
        self.selected_object_ids.clear()
        self.active_object_id = None
        self.clear_sub_selection()
        if self.edit_mode != EditMode.OBJECT:
            self.edit_mode = EditMode.OBJECT
        return before - len(self.objects)

    def duplicate_selected_objects(self) -> List[SceneObject]:
        selected = self.selected_objects()
        if not selected:
            return []
        self.push_undo("Duplicate Objects")
        created: List[SceneObject] = []
        new_ids: List[str] = []
        for obj in selected:
            dup = obj.clone(new_id=True)
            dup.name = self.unique_name(obj.name)
            dup.position[0] += 0.6
            self.objects.append(dup)
            created.append(dup)
            new_ids.append(dup.obj_id)
        self.selected_object_ids = new_ids
        self.active_object_id = new_ids[-1]
        self.clear_sub_selection()
        return created

    def active_edit_vertex_indices(self) -> Set[int]:
        """Return all vertex indices affected by the current Edit Mode selection."""
        obj = self.active_object()
        if obj is None:
            return set()
        mesh = obj.mesh
        if self.edit_mode == EditMode.VERTEX:
            return {v for v in self.selected_vertices if 0 <= v < mesh.vertex_count}
        if self.edit_mode == EditMode.EDGE:
            verts: Set[int] = set()
            for u, v in self.selected_edges:
                if 0 <= u < mesh.vertex_count:
                    verts.add(u)
                if 0 <= v < mesh.vertex_count:
                    verts.add(v)
            return verts
        if self.edit_mode == EditMode.FACE:
            verts = set()
            for fi in self.selected_faces:
                if 0 <= fi < mesh.face_count:
                    verts.update(mesh.faces[fi])
            return verts
        return set()

    def selection_pivot_world(self) -> Optional[np.ndarray]:
        """Compute the world-space pivot point for the transform gizmo."""
        if self.edit_mode == EditMode.OBJECT:
            selected = self.selected_objects()
            if not selected:
                return None
            pts = np.array([o.position for o in selected], dtype=np.float32)
            return pts.mean(axis=0).astype(np.float32)

        obj = self.active_object()
        if obj is None:
            return None
        v_indices = sorted(self.active_edit_vertex_indices())
        if not v_indices:
            return None
        world_v = obj.world_vertices()
        return world_v[v_indices].mean(axis=0).astype(np.float32)

    def select_all_toggle(self) -> None:
        if self.edit_mode == EditMode.OBJECT:
            if len(self.selected_object_ids) == len(self.objects) and self.objects:
                self.selected_object_ids.clear()
                self.active_object_id = None
            else:
                self.selected_object_ids = [o.obj_id for o in self.objects if o.visible]
                self.active_object_id = (
                    self.selected_object_ids[-1] if self.selected_object_ids else None
                )
            return

        obj = self.active_object()
        if obj is None:
            return
        mesh = obj.mesh
        if self.edit_mode == EditMode.VERTEX:
            if len(self.selected_vertices) == mesh.vertex_count:
                self.selected_vertices.clear()
            else:
                self.selected_vertices = set(range(mesh.vertex_count))
        elif self.edit_mode == EditMode.EDGE:
            all_edges = set(mesh.unique_edges())
            if len(self.selected_edges) == len(all_edges):
                self.selected_edges.clear()
            else:
                self.selected_edges = all_edges
        elif self.edit_mode == EditMode.FACE:
            if len(self.selected_faces) == mesh.face_count:
                self.selected_faces.clear()
            else:
                self.selected_faces = set(range(mesh.face_count))

    def join_selected_objects(self) -> Optional[SceneObject]:
        """Merge all selected objects into the active object's local coordinate space."""
        selected = [o for o in self.selected_objects() if not o.locked]
        if len(selected) < 2:
            return None
        target = self.active_object()
        if target is None or target.locked or target not in selected:
            target = selected[-1]

        self.push_undo("Join Objects")
        inv_target = np.linalg.inv(target.model_matrix().astype(np.float64)).astype(np.float32)

        all_verts = [target.mesh.vertices.copy()]
        all_faces = [list(f) for f in target.mesh.faces]
        v_offset = len(target.mesh.vertices)

        removed_ids: Set[str] = set()
        for obj in selected:
            if obj.obj_id == target.obj_id:
                continue
            world_v = obj.world_vertices()
            local_v = transform_points(world_v, inv_target)
            all_verts.append(local_v)
            for face in obj.mesh.faces:
                all_faces.append([vi + v_offset for vi in face])
            v_offset += len(local_v)
            removed_ids.add(obj.obj_id)

        target.mesh.vertices = np.vstack(all_verts).astype(np.float32)
        target.mesh.faces = all_faces
        target.mesh.touch()

        self.objects = [o for o in self.objects if o.obj_id not in removed_ids]
        self.selected_object_ids = [target.obj_id]
        self.active_object_id = target.obj_id
        self.edit_mode = EditMode.OBJECT
        self.clear_sub_selection()
        return target

    def select_edge_loop(self) -> int:
        """Expand currently selected edge(s) into full topological edge loops."""
        obj = self.active_object()
        if obj is None or self.edit_mode != EditMode.EDGE or not self.selected_edges:
            return 0
        expanded: Set[Edge] = set(self.selected_edges)
        for edge in list(self.selected_edges):
            expanded.update(obj.mesh.find_edge_loop(edge))
        self.selected_edges = expanded
        return len(expanded)

    def grow_selection(self) -> int:
        """Expand active Edit Mode selection by one neighbor ring."""
        obj = self.active_object()
        if obj is None or self.edit_mode == EditMode.OBJECT:
            return 0
        if self.edit_mode == EditMode.VERTEX and self.selected_vertices:
            self.selected_vertices = obj.mesh.grow_vertex_selection(self.selected_vertices)
            return len(self.selected_vertices)
        if self.edit_mode == EditMode.EDGE and self.selected_edges:
            self.selected_edges = obj.mesh.grow_edge_selection(self.selected_edges)
            return len(self.selected_edges)
        if self.edit_mode == EditMode.FACE and self.selected_faces:
            self.selected_faces = obj.mesh.grow_face_selection(self.selected_faces)
            return len(self.selected_faces)
        return 0

    def invert_selection(self) -> None:
        """Invert selection in Object Mode or Edit Mode."""
        if self.edit_mode == EditMode.OBJECT:
            cur = set(self.selected_object_ids)
            self.selected_object_ids = [o.obj_id for o in self.objects if o.visible and o.obj_id not in cur]
            self.active_object_id = self.selected_object_ids[-1] if self.selected_object_ids else None
            return
        obj = self.active_object()
        if obj is None:
            return
        if self.edit_mode == EditMode.VERTEX:
            self.selected_vertices = set(range(obj.mesh.vertex_count)) - self.selected_vertices
        elif self.edit_mode == EditMode.EDGE:
            self.selected_edges = set(obj.mesh.unique_edges()) - self.selected_edges
        elif self.edit_mode == EditMode.FACE:
            self.selected_faces = set(range(obj.mesh.face_count)) - self.selected_faces

    # ------------------------------------------------------------------
    # Project Serialization (.p3d JSON)
    # ------------------------------------------------------------------

    def save_project(self, path: str | Path) -> None:
        payload = {
            "format": "py3d_modeler_project",
            "version": 1,
            "objects": [o.to_dict() for o in self.objects],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_project(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.push_undo("Load Project")
        self.objects = [SceneObject.from_dict(o) for o in data.get("objects", [])]
        self.selected_object_ids = [self.objects[0].obj_id] if self.objects else []
        self.active_object_id = self.objects[0].obj_id if self.objects else None
        self.edit_mode = EditMode.OBJECT
        self.clear_sub_selection()
