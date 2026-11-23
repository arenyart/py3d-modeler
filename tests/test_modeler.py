"""Automated test suite for Py3D Modeler core math, mesh ops, scene, I/O, and Qt GUI."""

from __future__ import annotations

import numpy as np
import pytest

from modeler.io_formats import export_obj, export_ply, export_stl_binary, import_obj, import_stl
from modeler.math3d import (
    look_at,
    perspective,
    ray_aabb_intersect,
    ray_triangle_intersect,
    screen_to_ray,
    trs_matrix,
)
from modeler.primitives import PRIMITIVE_BUILDERS, create_cube, create_torus
from modeler.scene import EditMode, Scene, SceneObject, ShadingMode, TransformTool


# ---------------------------------------------------------------------------
# Math & Raycasting Tests
# ---------------------------------------------------------------------------

def test_ray_triangle_and_aabb():
    origin = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    v0 = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    v1 = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    t = ray_triangle_intersect(origin, direction, v0, v1, v2)
    assert t is not None and abs(t - 5.0) < 1e-4

    miss = ray_triangle_intersect(origin, np.array([0.0, 1.0, 0.0], dtype=np.float32), v0, v1, v2)
    assert miss is None

    t_box = ray_aabb_intersect(origin, direction, np.array([-1, -1, -1]), np.array([1, 1, 1]))
    assert t_box is not None and abs(t_box - 4.0) < 1e-4


def test_screen_to_ray_center():
    view = look_at(np.array([0, 0, 5]), np.array([0, 0, 0]), np.array([0, 1, 0]))
    proj = perspective(45.0, 1.0, 0.1, 100.0)
    origin, direction = screen_to_ray(400, 400, 800, 800, view, proj)
    assert np.allclose(direction, [0.0, 0.0, -1.0], atol=1e-3)
    assert origin[2] > 4.0


# ---------------------------------------------------------------------------
# Primitives & Mesh Modeling Operations Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(PRIMITIVE_BUILDERS.keys()))
def test_all_primitives_valid(name: str):
    mesh = PRIMITIVE_BUILDERS[name]()
    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
    assert mesh.edge_count > 0
    pos, nrm, face_ids = mesh.triangulated_arrays(smooth=False)
    assert len(pos) == mesh.triangle_count * 3
    assert len(nrm) == len(pos)
    assert len(face_ids) == len(pos)
    # All face normals should have unit length
    norms = np.linalg.norm(nrm, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_extrude_inset_subdivide_operations():
    cube = create_cube(2.0)
    assert cube.vertex_count == 8
    assert cube.face_count == 6

    # Extrude top face (+Y face index 4)
    caps = cube.extrude_faces([4], distance=1.0)
    assert caps == [4]
    assert cube.vertex_count == 12
    assert cube.face_count == 10
    assert abs(cube.face_center(4)[1] - 2.0) < 1e-4

    # Inset the extruded top face
    inset_caps = cube.inset_faces([4], factor=0.3)
    assert inset_caps == [4]
    assert cube.vertex_count == 16
    assert cube.face_count == 14

    # Catmull-Clark subdivision produces quads (4x faces for all-quad mesh)
    prev_faces = cube.face_count
    cube.subdivide_catmull_clark()
    assert cube.face_count == prev_faces * 4
    assert all(len(f) == 4 for f in cube.faces)

    # Triangulate turns all quads into 2 tris each
    quad_count = cube.face_count
    cube.triangulate()
    assert cube.face_count == quad_count * 2
    assert all(len(f) == 3 for f in cube.faces)


def test_simple_subdivide_and_delete():
    cube = create_cube(2.0)
    cube.subdivide_simple()
    assert cube.face_count == 24
    cube.delete_faces([0, 1])
    assert cube.face_count == 22


# ---------------------------------------------------------------------------
# Scene, Undo/Redo & Serialization Tests
# ---------------------------------------------------------------------------

def test_scene_undo_redo_and_project_io(tmp_path):
    scene = Scene()
    scene.reset_default_scene()
    assert len(scene.objects) == 1
    assert not scene.can_undo

    torus = SceneObject(name="Torus", mesh=create_torus(), position=[2.0, 1.0, 0.0])
    scene.add_object(torus)
    assert len(scene.objects) == 2
    assert scene.can_undo

    scene.duplicate_selected_objects()
    assert len(scene.objects) == 3

    scene.undo()
    assert len(scene.objects) == 2
    assert scene.can_redo

    scene.redo()
    assert len(scene.objects) == 3

    proj_file = tmp_path / "test.p3d"
    scene.save_project(proj_file)

    scene2 = Scene()
    scene2.load_project(proj_file)
    assert len(scene2.objects) == 3
    assert scene2.objects[0].name == "Cube"


# ---------------------------------------------------------------------------
# OBJ / STL / PLY Import & Export Tests
# ---------------------------------------------------------------------------

def test_obj_stl_ply_roundtrip(tmp_path):
    obj = SceneObject(name="HeroCube", mesh=create_cube(2.0), position=[1.0, 2.0, 3.0])

    obj_path = tmp_path / "hero.obj"
    export_obj(obj_path, [obj])
    imported_obj = import_obj(obj_path)
    assert len(imported_obj) == 1
    name, mesh_obj = imported_obj[0]
    assert name == "HeroCube"
    assert mesh_obj.vertex_count == 8
    assert mesh_obj.face_count == 6
    bmin, bmax = mesh_obj.compute_bounds()
    assert np.allclose((bmin + bmax) * 0.5, [1.0, 2.0, 3.0], atol=1e-4)

    stl_path = tmp_path / "hero.stl"
    export_stl_binary(stl_path, [obj])
    mesh_stl = import_stl(stl_path)
    assert mesh_stl.vertex_count == 8
    assert mesh_stl.face_count == 12  # triangulated cube

    ply_path = tmp_path / "hero.ply"
    export_ply(ply_path, [obj])
    ply_text = ply_path.read_text(encoding="utf-8")
    assert "element vertex 8" in ply_text
    assert "element face 6" in ply_text


# ---------------------------------------------------------------------------
# New Modeling Tools Tests
# ---------------------------------------------------------------------------

def test_bevel_cube_topology_and_outward_normals():
    cube = create_cube(2.0)
    cube.bevel(factor=0.22)
    # 6 original faces + 12 edge bevel quads + 8 corner triangles = 26 faces, 24 verts
    assert cube.vertex_count == 24
    assert cube.face_count == 26
    # Verify every face normal points outward (dot(face_center, face_normal) > 0)
    for fi in range(cube.face_count):
        center = cube.face_center(fi)
        normal = cube.face_normal(fi)
        assert float(np.dot(center, normal)) > 0.1


def test_bevel_selected_edges_only():
    # 1. Single edge bevel on a cube: front-top edge (2, 3)
    cube1 = create_cube(2.0)
    new_edges = cube1.bevel(factor=0.25, edge_indices={(2, 3)})
    assert len(new_edges) == 2
    # 6 original faces (2 end caps become pentagons) + 1 bevel quad = 7 faces, 10 verts
    assert cube1.face_count == 7
    assert cube1.vertex_count == 10
    for fi in range(cube1.face_count):
        assert float(np.dot(cube1.face_center(fi), cube1.face_normal(fi))) > 0.1

    # 2. Top ring bevel (all 4 edges of top face [3, 2, 6, 7]): clean mitered quad bevel!
    cube2 = create_cube(2.0)
    top_ring = {(3, 2), (2, 6), (6, 7), (7, 3)}
    new_ring_edges = cube2.bevel(factor=0.25, edge_indices=top_ring)
    assert len(new_ring_edges) == 8
    # 6 original faces + 4 bevel quads = 10 faces (all quads!), 12 verts (4 bottom + 4 upper-side + 4 top-inset)
    assert cube2.face_count == 10
    assert cube2.vertex_count == 12
    assert all(len(f) == 4 for f in cube2.faces)
    for fi in range(cube2.face_count):
        assert float(np.dot(cube2.face_center(fi), cube2.face_normal(fi))) > 0.1


def test_loop_cut_and_edge_loop_selection():
    cube = create_cube(2.0)
    # Seed edge along X on front face -> cuts the 4-quad ring around X
    new_edges = cube.loop_cut(seed_edge=(0, 1), cuts=2)
    assert len(new_edges) == 8  # 2 loops * 4 quads around ring
    assert cube.face_count == 6 - 4 + 4 * 3  # 14 faces (2 untouched end caps + 12 ring quads)

    # Edge loop expansion on one of the newly created cut edges should find all 4 edges in that ring
    seed = next(iter(new_edges))
    loop = cube.find_edge_loop(seed)
    assert len(loop) == 4


def test_poke_and_bridge_faces():
    cube = create_cube(2.0)
    # Poke top face with positive normal offset
    centers = cube.poke_faces([4], offset=0.8)
    assert len(centers) == 1
    assert cube.face_count == 9  # 5 quads + 4 triangles
    assert cube.vertices[centers[0]][1] > 1.7

    # Bridge two opposite faces of a fresh cube -> creates hollow tube (4 original sides + 4 bridge sides)
    box = create_cube(2.0)
    # Split into two disjoint cubes and bridge facing quads
    box2 = create_cube(2.0)
    box2.vertices[:, 0] += 4.0
    combined_v = np.vstack([box.vertices, box2.vertices])
    combined_f = [list(f) for f in box.faces] + [[vi + 8 for vi in f] for f in box2.faces]
    from modeler.mesh import Mesh
    m = Mesh(combined_v, combined_f)
    # Face 3 is +X of cube1; face 6+2=8 is -X of cube2
    assert m.bridge_faces(3, 8)
    assert m.face_count == 12 - 2 + 4


def test_solidify_mirror_and_radial_array():
    from modeler.primitives import create_plane
    plane = create_plane(size=2.0, subdivisions=2)
    orig_faces = plane.face_count
    plane.solidify(thickness=0.2)
    # 4 top + 4 bottom + 8 boundary rim quads = 16 faces
    assert plane.face_count == orig_faces * 2 + 8

    # Mirror shifted cube across X with seam touching X=0 -> welds 4 shared seam vertices
    half = create_cube(2.0)
    half.vertices[:, 0] += 1.0  # sits in x in [0, 2]
    half.mirror(axis=0, weld_threshold=1e-3)
    assert half.vertex_count == 12  # 8 + 8 - 4 welded seam verts
    assert half.face_count == 12

    # Radial array 6x around Y
    spoke = create_cube(0.5)
    spoke.radial_array(count=6, radius=2.0)
    assert spoke.vertex_count == 8 * 6
    assert spoke.face_count == 6 * 6


def test_deformers_and_cleanup():
    cube = create_cube(2.0)
    cube.subdivide_simple()
    before_verts = cube.vertices.copy()
    cube.smooth_vertices(factor=0.5, iterations=2)
    assert not np.allclose(cube.vertices, before_verts)

    bmin0, bmax0 = cube.compute_bounds()
    cube.inflate_vertices(distance=0.25)
    bmin1, bmax1 = cube.compute_bounds()
    assert float(np.linalg.norm(bmax1 - bmin1)) > float(np.linalg.norm(bmax0 - bmin0))

    cube.twist(angle_deg=90.0, axis=1)
    cube.taper(factor=-0.5, axis=1)
    assert cube.vertex_count > 0

    # Flip one face & test recalculate_normals fixes all faces to outward
    bad = create_cube(2.0)
    bad.flip_normals([0, 2, 4])
    bad.recalculate_normals()
    for fi in range(bad.face_count):
        assert float(np.dot(bad.face_center(fi), bad.face_normal(fi))) > 0.5

    # Weld duplicate vertices
    dup = create_cube(2.0)
    dup.vertices = np.vstack([dup.vertices, dup.vertices])
    dup.faces = dup.faces + [[v + 8 for v in f] for f in dup.faces]
    merged = dup.merge_by_distance(0.01)
    assert merged == 8
    assert dup.vertex_count == 8


def test_join_objects_and_selection_helpers():
    scene = Scene()
    scene.reset_default_scene()
    c2 = SceneObject(name="Cube2", mesh=create_cube(1.0), position=[3.0, 0.5, 0.0])
    scene.add_object(c2)
    scene.selected_object_ids = [o.obj_id for o in scene.objects]
    scene.active_object_id = scene.objects[0].obj_id

    joined = scene.join_selected_objects()
    assert joined is not None
    assert len(scene.objects) == 1
    assert joined.mesh.face_count == 12
    assert joined.mesh.vertex_count == 16

    # Edit mode grow & invert selection
    scene.set_edit_mode(EditMode.FACE)
    scene.selected_faces = {0}
    grew = scene.grow_selection()
    assert grew > 1
    prev_len = len(scene.selected_faces)
    scene.invert_selection()
    assert len(scene.selected_faces) == joined.mesh.face_count - prev_len

