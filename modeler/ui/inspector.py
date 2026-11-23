"""Inspector panel for Object Transforms, Materials, and Polygonal Modeling Tools."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..scene import EditMode, MATERIAL_PRESETS, Scene


def _make_spin(
    minimum: float = -1000.0,
    maximum: float = 1000.0,
    step: float = 0.1,
    decimals: int = 2,
) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(minimum, maximum)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setKeyboardTracking(False)
    sb.setMinimumWidth(58)
    sb.setFixedHeight(25)
    sb.setAlignment(Qt.AlignmentFlag.AlignRight)
    return sb


def _make_int_spin(minimum: int = 1, maximum: int = 48, value: int = 1) -> QSpinBox:
    sb = QSpinBox()
    sb.setRange(minimum, maximum)
    sb.setValue(value)
    sb.setKeyboardTracking(False)
    sb.setMinimumWidth(52)
    sb.setFixedHeight(25)
    sb.setAlignment(Qt.AlignmentFlag.AlignRight)
    return sb


class InspectorPanel(QWidget):
    """Properties and modeling tools panel for the active SceneObject."""

    scene_modified = Signal()
    status_message = Signal(str)

    def __init__(self, scene: Scene, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._syncing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 6, 8, 10)
        self._layout.setSpacing(8)

        # Empty state label
        self.empty_label = QLabel("Select an object to inspect\nand edit its properties.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #8e8e8e; padding: 24px 8px;")
        self._layout.addWidget(self.empty_label)

        # Content container shown when an object is active
        self.content = QWidget()
        c_layout = QVBoxLayout(self.content)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(8)
        self._layout.addWidget(self.content)

        # --- Object Identity ---
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._on_name_changed)
        id_row.addWidget(self.name_edit)
        c_layout.addLayout(id_row)

        # --- Transform Group ---
        tf_group = QGroupBox("Transform")
        tf_grid = QGridLayout(tf_group)
        tf_grid.setSpacing(5)

        axis_colors = [("#eb5757", "X"), ("#58ce72", "Y"), ("#5298f5", "Z")]
        for col, (hex_col, label) in enumerate(axis_colors, start=1):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"color: {hex_col}; font-weight: 700; font-size: 11px;")
            tf_grid.addWidget(lbl, 0, col)

        self.pos_spins: List[QDoubleSpinBox] = []
        self.rot_spins: List[QDoubleSpinBox] = []
        self.scl_spins: List[QDoubleSpinBox] = []

        rows = [
            ("Position", self.pos_spins, -500.0, 500.0, 0.1),
            ("Rotation", self.rot_spins, -3600.0, 3600.0, 5.0),
            ("Scale", self.scl_spins, 0.01, 200.0, 0.1),
        ]
        for r_idx, (title, target_list, lo, hi, step) in enumerate(rows, start=1):
            tf_grid.addWidget(QLabel(title), r_idx, 0)
            for c_idx in range(3):
                sb = _make_spin(lo, hi, step, 2)
                sb.valueChanged.connect(self._on_transform_edited)
                target_list.append(sb)
                tf_grid.addWidget(sb, r_idx, c_idx + 1)

        reset_tf_btn = QPushButton("Reset Transform")
        reset_tf_btn.clicked.connect(self._reset_transform)
        tf_grid.addWidget(reset_tf_btn, 4, 0, 1, 4)
        c_layout.addWidget(tf_group)

        # --- Material Group ---
        mat_group = QGroupBox("Material")
        mat_grid = QGridLayout(mat_group)
        mat_grid.setSpacing(5)

        mat_grid.addWidget(QLabel("Preset"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom")
        for preset_name in MATERIAL_PRESETS:
            self.preset_combo.addItem(preset_name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        mat_grid.addWidget(self.preset_combo, 0, 1, 1, 2)

        mat_grid.addWidget(QLabel("Base Color"), 1, 0)
        self.color_btn = QPushButton("")
        self.color_btn.setFixedHeight(22)
        self.color_btn.clicked.connect(self._pick_color)
        mat_grid.addWidget(self.color_btn, 1, 1, 1, 2)

        mat_grid.addWidget(QLabel("Roughness"), 2, 0)
        self.rough_slider = QSlider(Qt.Orientation.Horizontal)
        self.rough_slider.setRange(5, 100)
        self.rough_slider.valueChanged.connect(self._on_material_slider)
        self.rough_val = QLabel("0.45")
        self.rough_val.setFixedWidth(32)
        mat_grid.addWidget(self.rough_slider, 2, 1)
        mat_grid.addWidget(self.rough_val, 2, 2)

        mat_grid.addWidget(QLabel("Metallic"), 3, 0)
        self.metal_slider = QSlider(Qt.Orientation.Horizontal)
        self.metal_slider.setRange(0, 100)
        self.metal_slider.valueChanged.connect(self._on_material_slider)
        self.metal_val = QLabel("0.15")
        self.metal_val.setFixedWidth(32)
        mat_grid.addWidget(self.metal_slider, 3, 1)
        mat_grid.addWidget(self.metal_val, 3, 2)
        c_layout.addWidget(mat_group)

        # --- Mesh Modeling Tools Group (Tabbed) ---
        mesh_group = QGroupBox("Mesh Modeling")
        m_layout = QVBoxLayout(mesh_group)
        m_layout.setSpacing(6)

        self.stats_label = QLabel("Verts: 0  ·  Edges: 0  ·  Faces: 0  ·  Tris: 0")
        self.stats_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        m_layout.addWidget(self.stats_label)

        self.tools_tabs = QTabWidget()
        m_layout.addWidget(self.tools_tabs)

        # ---- Tab 1: Edit (Polygonal Topology Tools) ----
        edit_tab = QWidget()
        e_lay = QVBoxLayout(edit_tab)
        e_lay.setContentsMargins(6, 8, 6, 8)
        e_lay.setSpacing(6)

        # Extrude
        ext_row = QHBoxLayout()
        self.extrude_dist = _make_spin(-20.0, 20.0, 0.1, 2)
        self.extrude_dist.setValue(0.50)
        self.extrude_btn = QPushButton("Extrude Faces")
        self.extrude_btn.setObjectName("AccentBtn")
        self.extrude_btn.setToolTip("Extrude selected faces (E) along their average normal")
        self.extrude_btn.clicked.connect(self.action_extrude_faces)
        ext_row.addWidget(self.extrude_btn, 2)
        ext_row.addWidget(self.extrude_dist, 1)
        e_lay.addLayout(ext_row)

        # Inset
        ins_row = QHBoxLayout()
        self.inset_factor = _make_spin(0.05, 0.90, 0.05, 2)
        self.inset_factor.setValue(0.25)
        self.inset_btn = QPushButton("Inset Faces")
        self.inset_btn.setObjectName("AccentBtn")
        self.inset_btn.setToolTip("Inset selected faces toward their centroid (I)")
        self.inset_btn.clicked.connect(self.action_inset_faces)
        ins_row.addWidget(self.inset_btn, 2)
        ins_row.addWidget(self.inset_factor, 1)
        e_lay.addLayout(ins_row)

        # Bevel
        bev_row = QHBoxLayout()
        self.bevel_factor = _make_spin(0.05, 0.45, 0.04, 2)
        self.bevel_factor.setValue(0.20)
        btn_bevel = QPushButton("Bevel / Chamfer")
        btn_bevel.setObjectName("PrimaryBtn")
        btn_bevel.setToolTip("Bevel selected edges in Edge Mode, or whole object in Object Mode (Ctrl+B)")
        btn_bevel.clicked.connect(self.action_bevel)
        bev_row.addWidget(btn_bevel, 2)
        bev_row.addWidget(self.bevel_factor, 1)
        e_lay.addLayout(bev_row)

        # Loop Cut
        lc_row = QHBoxLayout()
        self.loop_cuts = _make_int_spin(1, 12, 1)
        btn_loop = QPushButton("Loop Cut")
        btn_loop.setObjectName("PrimaryBtn")
        btn_loop.setToolTip("Slice quad edge ring with parallel cuts (Ctrl+R)")
        btn_loop.clicked.connect(self.action_loop_cut)
        lc_row.addWidget(btn_loop, 2)
        lc_row.addWidget(self.loop_cuts, 1)
        e_lay.addLayout(lc_row)

        # Poke Faces
        poke_row = QHBoxLayout()
        self.poke_offset = _make_spin(-5.0, 5.0, 0.1, 2)
        self.poke_offset.setValue(0.25)
        btn_poke = QPushButton("Poke Faces")
        btn_poke.setToolTip("Insert center vertex into selected faces with normal offset")
        btn_poke.clicked.connect(self.action_poke_faces)
        poke_row.addWidget(btn_poke, 2)
        poke_row.addWidget(self.poke_offset, 1)
        e_lay.addLayout(poke_row)

        # Bridge & Selection Helpers
        edit_grid = QGridLayout()
        edit_grid.setSpacing(5)
        btn_bridge = QPushButton("Bridge 2 Faces")
        btn_bridge.setToolTip("Connect 2 selected faces with a quad tunnel")
        btn_bridge.clicked.connect(self.action_bridge_faces)

        btn_del_sel = QPushButton("Delete Elements")
        btn_del_sel.setToolTip("Delete selected vertices or faces in Edit Mode")
        btn_del_sel.clicked.connect(self.action_delete_elements)

        btn_sel_loop = QPushButton("Edge Loop (L)")
        btn_sel_loop.setToolTip("Expand selected edge to its full topological loop")
        btn_sel_loop.clicked.connect(self.action_select_edge_loop)

        btn_grow = QPushButton("Grow Select (+)")
        btn_grow.setToolTip("Expand vertex, edge, or face selection by one ring")
        btn_grow.clicked.connect(self.action_grow_selection)

        edit_grid.addWidget(btn_bridge, 0, 0)
        edit_grid.addWidget(btn_del_sel, 0, 1)
        edit_grid.addWidget(btn_sel_loop, 1, 0)
        edit_grid.addWidget(btn_grow, 1, 1)
        e_lay.addLayout(edit_grid)

        self.tools_tabs.addTab(edit_tab, "Edit")

        # ---- Tab 2: Modify (Subdivide, Solidify, Mirror, Array, Join) ----
        mod_tab = QWidget()
        mod_lay = QVBoxLayout(mod_tab)
        mod_lay.setContentsMargins(6, 8, 6, 8)
        mod_lay.setSpacing(6)

        sub_row = QHBoxLayout()
        btn_cc = QPushButton("Catmull-Clark")
        btn_cc.setObjectName("PrimaryBtn")
        btn_cc.setToolTip("Apply one level of Catmull-Clark smooth subdivision")
        btn_cc.clicked.connect(self.action_catmull_clark)
        btn_sub = QPushButton("Subdivide")
        btn_sub.setToolTip("Subdivide faces into quads without smoothing")
        btn_sub.clicked.connect(self.action_subdivide_simple)
        sub_row.addWidget(btn_cc)
        sub_row.addWidget(btn_sub)
        mod_lay.addLayout(sub_row)

        # Solidify
        sol_row = QHBoxLayout()
        self.solidify_thick = _make_spin(-5.0, 5.0, 0.05, 2)
        self.solidify_thick.setValue(0.18)
        btn_sol = QPushButton("Solidify Shell")
        btn_sol.setToolTip("Add wall thickness to thin or open meshes")
        btn_sol.clicked.connect(self.action_solidify)
        sol_row.addWidget(btn_sol, 2)
        sol_row.addWidget(self.solidify_thick, 1)
        mod_lay.addLayout(sol_row)

        # Mirror
        mir_row = QHBoxLayout()
        self.mirror_axis = QComboBox()
        self.mirror_axis.addItems(["X Axis", "Y Axis", "Z Axis"])
        self.mirror_axis.setFixedHeight(25)
        btn_mir = QPushButton("Mirror & Weld")
        btn_mir.setToolTip("Mirror geometry across chosen local axis and weld seam")
        btn_mir.clicked.connect(self.action_mirror)
        mir_row.addWidget(btn_mir, 2)
        mir_row.addWidget(self.mirror_axis, 1)
        mod_lay.addLayout(mir_row)

        # Radial Array
        arr_row = QHBoxLayout()
        self.array_count = _make_int_spin(2, 36, 6)
        self.array_radius = _make_spin(-20.0, 20.0, 0.25, 2)
        self.array_radius.setValue(1.50)
        self.array_radius.setToolTip("Radial offset distance before rotating")
        btn_arr = QPushButton("Radial Array")
        btn_arr.setToolTip("Duplicate geometry radially around Y axis (count, radius)")
        btn_arr.clicked.connect(self.action_radial_array)
        arr_row.addWidget(btn_arr, 2)
        arr_row.addWidget(self.array_count, 1)
        arr_row.addWidget(self.array_radius, 1)
        mod_lay.addLayout(arr_row)

        # Join Selected Objects
        btn_join = QPushButton("Join Selected Objects (Ctrl+J)")
        btn_join.setToolTip("Merge 2+ selected objects into a single editable mesh")
        btn_join.clicked.connect(self.action_join_objects)
        mod_lay.addWidget(btn_join)

        self.tools_tabs.addTab(mod_tab, "Modify")

        # ---- Tab 3: Deform & Fix (Smooth, Inflate, Twist, Taper, Cleanup) ----
        def_tab = QWidget()
        d_lay = QVBoxLayout(def_tab)
        d_lay.setContentsMargins(6, 8, 6, 8)
        d_lay.setSpacing(6)

        # Laplacian Smooth
        sm_row = QHBoxLayout()
        self.smooth_factor = _make_spin(0.05, 1.0, 0.1, 2)
        self.smooth_factor.setValue(0.50)
        btn_sm = QPushButton("Smooth Verts")
        btn_sm.setToolTip("Laplacian smooth selected vertices (or whole mesh)")
        btn_sm.clicked.connect(self.action_smooth)
        sm_row.addWidget(btn_sm, 2)
        sm_row.addWidget(self.smooth_factor, 1)
        d_lay.addLayout(sm_row)

        # Inflate / Shrink
        inf_row = QHBoxLayout()
        self.inflate_dist = _make_spin(-5.0, 5.0, 0.05, 2)
        self.inflate_dist.setValue(0.15)
        btn_inf = QPushButton("Inflate / Shrink")
        btn_inf.setToolTip("Push vertices outward (+) or inward (-) along normals")
        btn_inf.clicked.connect(self.action_inflate)
        inf_row.addWidget(btn_inf, 2)
        inf_row.addWidget(self.inflate_dist, 1)
        d_lay.addLayout(inf_row)

        # Twist
        tw_row = QHBoxLayout()
        self.twist_deg = _make_spin(-720.0, 720.0, 15.0, 1)
        self.twist_deg.setValue(90.0)
        btn_tw = QPushButton("Twist (Y°)")
        btn_tw.setToolTip("Twist mesh progressively along vertical Y height")
        btn_tw.clicked.connect(self.action_twist)
        tw_row.addWidget(btn_tw, 2)
        tw_row.addWidget(self.twist_deg, 1)
        d_lay.addLayout(tw_row)

        # Taper
        tap_row = QHBoxLayout()
        self.taper_factor = _make_spin(-2.0, 2.0, 0.1, 2)
        self.taper_factor.setValue(-0.60)
        btn_tap = QPushButton("Taper (Y)")
        btn_tap.setToolTip("Linearly narrow (-) or flare (+) top relative to bottom")
        btn_tap.clicked.connect(self.action_taper)
        tap_row.addWidget(btn_tap, 2)
        tap_row.addWidget(self.taper_factor, 1)
        d_lay.addLayout(tap_row)

        # Cleanup & Normals Grid
        fix_grid = QGridLayout()
        fix_grid.setSpacing(5)
        btn_tri = QPushButton("Triangulate")
        btn_tri.clicked.connect(self.action_triangulate)

        btn_flip = QPushButton("Flip Normals")
        btn_flip.clicked.connect(self.action_flip_normals)

        btn_recalc = QPushButton("Fix Normals")
        btn_recalc.setToolTip("Recalculate consistent outward-facing normals")
        btn_recalc.clicked.connect(self.action_recalculate_normals)

        btn_weld = QPushButton("Weld Verts")
        btn_weld.setToolTip("Merge duplicate vertices within 0.01 distance")
        btn_weld.clicked.connect(self.action_merge_by_distance)

        btn_center = QPushButton("Center Pivot")
        btn_center.clicked.connect(self.action_center_pivot)

        fix_grid.addWidget(btn_tri, 0, 0)
        fix_grid.addWidget(btn_flip, 0, 1)
        fix_grid.addWidget(btn_recalc, 1, 0)
        fix_grid.addWidget(btn_weld, 1, 1)
        fix_grid.addWidget(btn_center, 2, 0, 1, 2)
        d_lay.addLayout(fix_grid)

        self.tools_tabs.addTab(def_tab, "Deform")

        c_layout.addWidget(mesh_group)
        self._layout.addStretch(1)

    # ------------------------------------------------------------------
    # Refresh / Sync
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        obj = self.scene.active_object()
        has_obj = obj is not None
        self.empty_label.setVisible(not has_obj)
        self.content.setVisible(has_obj)
        if obj is None:
            return

        self._syncing = True
        self.name_edit.setText(obj.name)

        for i in range(3):
            self.pos_spins[i].setValue(float(obj.position[i]))
            self.rot_spins[i].setValue(float(obj.rotation[i]))
            self.scl_spins[i].setValue(float(obj.scale[i]))

        r, g, b = [int(max(0, min(255, round(c * 255)))) for c in obj.material.color]
        self.color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #48536a; border-radius: 4px;"
        )
        self.rough_slider.setValue(int(round(obj.material.roughness * 100)))
        self.rough_val.setText(f"{obj.material.roughness:.2f}")
        self.metal_slider.setValue(int(round(obj.material.metallic * 100)))
        self.metal_val.setText(f"{obj.material.metallic:.2f}")

        idx = self.preset_combo.findText(obj.material.name)
        self.preset_combo.setCurrentIndex(idx if idx >= 0 else 0)

        m = obj.mesh
        self.stats_label.setText(
            f"Verts: {m.vertex_count:,}  ·  Edges: {m.edge_count:,}  ·  Faces: {m.face_count:,}  ·  Tris: {m.triangle_count:,}"
        )
        self._syncing = False

    # ------------------------------------------------------------------
    # Property Callbacks
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        if self._syncing:
            return
        obj = self.scene.active_object()
        if obj is None:
            return
        new_name = self.name_edit.text().strip()
        if new_name and new_name != obj.name:
            self.scene.push_undo("Rename Object")
            obj.name = new_name
            self.scene_modified.emit()

    def _on_transform_edited(self) -> None:
        if self._syncing:
            return
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Edit Transform")
        obj.position = [sb.value() for sb in self.pos_spins]
        obj.rotation = [sb.value() for sb in self.rot_spins]
        obj.scale = [sb.value() for sb in self.scl_spins]
        self.scene_modified.emit()

    def _reset_transform(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Reset Transform")
        obj.position = [0.0, 0.0, 0.0]
        obj.rotation = [0.0, 0.0, 0.0]
        obj.scale = [1.0, 1.0, 1.0]
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Reset transform for {obj.name}")

    def _pick_color(self) -> None:
        obj = self.scene.active_object()
        if obj is None:
            return
        r, g, b = [int(c * 255) for c in obj.material.color]
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Select Material Color")
        if chosen.isValid():
            self.scene.push_undo("Change Material Color")
            obj.material.color = (chosen.redF(), chosen.greenF(), chosen.blueF())
            obj.material.name = "Custom"
            self.refresh()
            self.scene_modified.emit()

    def _on_preset_selected(self, text: str) -> None:
        if self._syncing or text not in MATERIAL_PRESETS:
            return
        obj = self.scene.active_object()
        if obj is None:
            return
        self.scene.push_undo(f"Apply Material {text}")
        obj.material = MATERIAL_PRESETS[text].clone()
        self.refresh()
        self.scene_modified.emit()

    def _on_material_slider(self) -> None:
        if self._syncing:
            return
        obj = self.scene.active_object()
        if obj is None:
            return
        obj.material.roughness = self.rough_slider.value() / 100.0
        obj.material.metallic = self.metal_slider.value() / 100.0
        self.rough_val.setText(f"{obj.material.roughness:.2f}")
        self.metal_val.setText(f"{obj.material.metallic:.2f}")
        self.scene_modified.emit()

    # ------------------------------------------------------------------
    # Mesh Modeling Actions — Edit Tab
    # ------------------------------------------------------------------

    def _target_faces(self) -> Optional[List[int]]:
        obj = self.scene.active_object()
        if obj is None:
            return None
        if self.scene.edit_mode == EditMode.FACE and self.scene.selected_faces:
            return sorted(self.scene.selected_faces)
        return list(range(obj.mesh.face_count))

    def _target_vertices(self) -> Optional[List[int]]:
        if self.scene.edit_mode != EditMode.OBJECT:
            verts = sorted(self.scene.active_edit_vertex_indices())
            if verts:
                return verts
        return None

    def action_extrude_faces(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        faces = self._target_faces()
        if not faces:
            self.status_message.emit("No faces available to extrude.")
            return
        dist = float(self.extrude_dist.value())
        self.scene.push_undo("Extrude Faces")
        caps = obj.mesh.extrude_faces(faces, distance=dist)
        if self.scene.edit_mode == EditMode.FACE:
            self.scene.selected_faces = set(caps)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Extruded {len(caps)} face(s) by {dist:.2f}")

    def action_inset_faces(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        faces = self._target_faces()
        if not faces:
            self.status_message.emit("No faces available to inset.")
            return
        factor = float(self.inset_factor.value())
        self.scene.push_undo("Inset Faces")
        caps = obj.mesh.inset_faces(faces, factor=factor)
        if self.scene.edit_mode == EditMode.FACE:
            self.scene.selected_faces = set(caps)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Inset {len(caps)} face(s) by {factor:.2f}")

    def action_bevel(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        factor = float(self.bevel_factor.value())

        if self.scene.edit_mode == EditMode.EDGE:
            if not self.scene.selected_edges:
                self.status_message.emit("Select one or more edges in Edge Mode to bevel.")
                return
            edges_to_bevel = set(self.scene.selected_edges)
            self.scene.push_undo("Bevel Selected Edges")
            new_edges = obj.mesh.bevel(factor=factor, edge_indices=edges_to_bevel)
            if not new_edges:
                self.scene.undo()
                self.status_message.emit("Selected edge(s) must be shared by two faces to bevel.")
                return
            self.scene.selected_edges = new_edges
            self.refresh()
            self.scene_modified.emit()
            self.status_message.emit(
                f"Beveled {len(edges_to_bevel)} edge(s) (factor {factor:.2f}, {obj.mesh.face_count} faces)"
            )
            return

        if obj.mesh.face_count > 10000:
            self.status_message.emit("Mesh too dense for full bevel (>10k faces).")
            return
        self.scene.push_undo("Bevel Object")
        obj.mesh.bevel(factor=factor, edge_indices=None)
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Beveled {obj.name} (factor {factor:.2f}, {obj.mesh.face_count} faces)")

    def action_loop_cut(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        cuts = int(self.loop_cuts.value())
        seed = next(iter(self.scene.selected_edges), None) if self.scene.edit_mode == EditMode.EDGE else None
        self.scene.push_undo("Loop Cut")
        new_edges = obj.mesh.loop_cut(seed_edge=seed, cuts=cuts)
        if not new_edges:
            self.scene.undo()
            self.status_message.emit("Loop Cut requires quad faces.")
            return
        if self.scene.edit_mode == EditMode.EDGE:
            self.scene.selected_edges = new_edges
        else:
            self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Added {cuts} loop cut(s) ({len(new_edges)} edges)")

    def action_poke_faces(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        faces = self._target_faces()
        if not faces:
            return
        offset = float(self.poke_offset.value())
        self.scene.push_undo("Poke Faces")
        centers = obj.mesh.poke_faces(faces, offset=offset)
        self.scene.clear_sub_selection()
        if self.scene.edit_mode == EditMode.VERTEX:
            self.scene.selected_vertices = set(centers)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Poked {len(centers)} face(s) (offset {offset:.2f})")

    def action_bridge_faces(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        if self.scene.edit_mode != EditMode.FACE or len(self.scene.selected_faces) != 2:
            self.status_message.emit("Select exactly 2 faces (with equal vertex count) in Face Mode to bridge.")
            return
        fa, fb = sorted(self.scene.selected_faces)
        self.scene.push_undo("Bridge Faces")
        ok = obj.mesh.bridge_faces(fa, fb)
        if not ok:
            self.scene.undo()
            self.status_message.emit("Cannot bridge: selected faces must have the same number of vertices.")
            return
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Bridged faces {fa} & {fb}")

    def action_select_edge_loop(self) -> None:
        count = self.scene.select_edge_loop()
        if count == 0:
            self.status_message.emit("Switch to Edge Mode and select an edge first.")
            return
        self.scene_modified.emit()
        self.status_message.emit(f"Selected edge loop ({count} edges)")

    def action_grow_selection(self) -> None:
        count = self.scene.grow_selection()
        if count == 0:
            self.status_message.emit("Select an element in Edit Mode to grow selection.")
            return
        self.scene_modified.emit()
        self.status_message.emit(f"Grew selection ({count} elements)")

    def action_delete_elements(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        if self.scene.edit_mode == EditMode.FACE and self.scene.selected_faces:
            self.scene.push_undo("Delete Faces")
            obj.mesh.delete_faces(self.scene.selected_faces)
            self.scene.clear_sub_selection()
        elif self.scene.edit_mode == EditMode.VERTEX and self.scene.selected_vertices:
            self.scene.push_undo("Delete Vertices")
            obj.mesh.delete_vertices(self.scene.selected_vertices)
            self.scene.clear_sub_selection()
        else:
            self.status_message.emit("Switch to Vertex or Face Edit Mode and select elements to delete.")
            return
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Deleted selected elements from {obj.name}")

    # ------------------------------------------------------------------
    # Mesh Modeling Actions — Modify Tab
    # ------------------------------------------------------------------

    def action_catmull_clark(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        if obj.mesh.face_count > 12000:
            self.status_message.emit("Mesh too dense for another Catmull-Clark level (>12k faces).")
            return
        self.scene.push_undo("Catmull-Clark Subdivide")
        obj.mesh.subdivide_catmull_clark()
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Catmull-Clark subdivided {obj.name} ({obj.mesh.face_count} faces)")

    def action_subdivide_simple(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        if obj.mesh.face_count > 12000:
            self.status_message.emit("Mesh too dense for another subdivision level (>12k faces).")
            return
        self.scene.push_undo("Simple Subdivide")
        obj.mesh.subdivide_simple()
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Subdivided {obj.name} ({obj.mesh.face_count} faces)")

    def action_solidify(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        thick = float(self.solidify_thick.value())
        self.scene.push_undo("Solidify")
        obj.mesh.solidify(thickness=thick)
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Solidified {obj.name} (thickness {thick:.2f})")

    def action_mirror(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        axis = self.mirror_axis.currentIndex()
        axis_name = ("X", "Y", "Z")[axis]
        self.scene.push_undo(f"Mirror {axis_name}")
        obj.mesh.mirror(axis=axis)
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Mirrored {obj.name} across {axis_name} axis")

    def action_radial_array(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        count = int(self.array_count.value())
        radius = float(self.array_radius.value())
        if obj.mesh.face_count * count > 30000:
            self.status_message.emit("Array result too dense (>30k faces).")
            return
        self.scene.push_undo("Radial Array")
        obj.mesh.radial_array(count=count, radius=radius)
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Created {count}x radial array (radius {radius:.2f})")

    def action_join_objects(self) -> None:
        joined = self.scene.join_selected_objects()
        if joined is None:
            self.status_message.emit("Select 2 or more objects (Shift+Click) to join them.")
            return
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Joined objects into {joined.name} ({joined.mesh.face_count} faces)")

    # ------------------------------------------------------------------
    # Mesh Modeling Actions — Deform & Fix Tab
    # ------------------------------------------------------------------

    def action_smooth(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        factor = float(self.smooth_factor.value())
        targets = self._target_vertices()
        self.scene.push_undo("Smooth Vertices")
        obj.mesh.smooth_vertices(vert_indices=targets, factor=factor, iterations=2)
        self.refresh()
        self.scene_modified.emit()
        scope = f"{len(targets)} selected verts" if targets else "entire mesh"
        self.status_message.emit(f"Smoothed {scope} (factor {factor:.2f})")

    def action_inflate(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        dist = float(self.inflate_dist.value())
        targets = self._target_vertices()
        self.scene.push_undo("Inflate / Shrink")
        obj.mesh.inflate_vertices(distance=dist, vert_indices=targets)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Inflated {obj.name} by {dist:+.2f}")

    def action_twist(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        deg = float(self.twist_deg.value())
        self.scene.push_undo("Twist Mesh")
        obj.mesh.twist(angle_deg=deg, axis=1)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Twisted {obj.name} by {deg:.1f}°")

    def action_taper(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        factor = float(self.taper_factor.value())
        self.scene.push_undo("Taper Mesh")
        obj.mesh.taper(factor=factor, axis=1)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Tapered {obj.name} (factor {factor:+.2f})")

    def action_triangulate(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Triangulate")
        target = (
            sorted(self.scene.selected_faces)
            if self.scene.edit_mode == EditMode.FACE and self.scene.selected_faces
            else None
        )
        obj.mesh.triangulate(target)
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Triangulated {obj.name}")

    def action_flip_normals(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Flip Normals")
        target = (
            sorted(self.scene.selected_faces)
            if self.scene.edit_mode == EditMode.FACE and self.scene.selected_faces
            else None
        )
        obj.mesh.flip_normals(target)
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Flipped normals on {obj.name}")

    def action_recalculate_normals(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Recalculate Normals")
        obj.mesh.recalculate_normals()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Recalculated outward normals on {obj.name}")

    def action_merge_by_distance(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Weld Vertices")
        merged = obj.mesh.merge_by_distance(0.01)
        self.scene.clear_sub_selection()
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Welded {merged} duplicate vertex/vertices on {obj.name}")

    def action_center_pivot(self) -> None:
        obj = self.scene.active_object()
        if obj is None or obj.locked:
            return
        self.scene.push_undo("Center Pivot")
        offset = obj.mesh.center_geometry()
        obj.position = [
            obj.position[0] + float(offset[0]) * obj.scale[0],
            obj.position[1] + float(offset[1]) * obj.scale[1],
            obj.position[2] + float(offset[2]) * obj.scale[2],
        ]
        self.refresh()
        self.scene_modified.emit()
        self.status_message.emit(f"Centered geometry pivot for {obj.name}")
