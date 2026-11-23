"""Main application window combining menus, toolbars, 3D viewport, outliner, and inspector."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QWidget,
)

from ..io_formats import export_obj, export_ply, export_stl_binary, import_obj, import_stl
from ..primitives import PRIMITIVE_BUILDERS
from ..scene import (
    EditMode,
    MATERIAL_PRESETS,
    Scene,
    SceneObject,
    ShadingMode,
    TransformTool,
)
from ..viewport import Viewport3D
from .inspector import InspectorPanel
from .outliner import OutlinerPanel
from .theme import DARK_STYLESHEET, make_icon


class MainWindow(QMainWindow):
    """Top-level studio window for Py3D Modeler."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Py3D Modeler — Python + PySide6")
        self.resize(1380, 860)
        self.setStyleSheet(DARK_STYLESHEET)

        self.scene = Scene()
        self.scene.reset_default_scene()
        self._current_project_path: Optional[Path] = None

        # Central 3D Viewport
        self.viewport = Viewport3D(self.scene, self)
        self.setCentralWidget(self.viewport)

        # Dock panels
        self.outliner = OutlinerPanel(self.scene, self)
        self.inspector = InspectorPanel(self.scene, self)

        self._outliner_dock = QDockWidget("Scene Outliner", self)
        self._outliner_dock.setWidget(self.outliner)
        self._outliner_dock.setMinimumWidth(300)
        self._outliner_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._outliner_dock)

        self._inspector_dock = QDockWidget("Inspector && Modeling", self)
        self._inspector_dock.setWidget(self.inspector)
        self._inspector_dock.setMinimumWidth(300)
        self._inspector_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._inspector_dock)
        self.splitDockWidget(self._outliner_dock, self._inspector_dock, Qt.Orientation.Vertical)
        self.resizeDocks([self._outliner_dock, self._inspector_dock], [190, 620], Qt.Orientation.Vertical)

        # Action registries
        self._tool_actions: Dict[TransformTool, QAction] = {}
        self._mode_actions: Dict[EditMode, QAction] = {}
        self._shading_actions: Dict[ShadingMode, QAction] = {}

        self._build_menus_and_toolbars()

        # Status bar
        self.status_label = QLabel("Ready — Py3D Modeler")
        self.statusBar().addWidget(self.status_label, 1)

        # Connect signals
        self.viewport.selection_changed.connect(self._sync_all_ui)
        self.viewport.scene_modified.connect(self._sync_all_ui)
        self.viewport.status_message.connect(self._show_status)

        self.outliner.selection_changed.connect(self._sync_all_ui)
        self.outliner.scene_modified.connect(self._sync_all_ui)
        self.outliner.focus_requested.connect(self.viewport.frame_selection)

        self.inspector.scene_modified.connect(self._sync_all_ui)
        self.inspector.status_message.connect(self._show_status)

        self._sync_all_ui()

    # ------------------------------------------------------------------
    # Menus & Toolbars Construction
    # ------------------------------------------------------------------

    def _build_menus_and_toolbars(self) -> None:
        menubar = self.menuBar()

        # --- File Menu ---
        file_menu = menubar.addMenu("&File")
        act_new = file_menu.addAction("New Scene", self._new_scene)
        act_new.setShortcut(QKeySequence.StandardKey.New)

        act_open = file_menu.addAction("Open Project (.p3d)...", self._open_project)
        act_open.setShortcut(QKeySequence.StandardKey.Open)

        act_save = file_menu.addAction("Save Project (.p3d)", self._save_project)
        act_save.setShortcut(QKeySequence.StandardKey.Save)

        file_menu.addAction("Save Project As...", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Import OBJ / STL...", self._import_mesh)
        file_menu.addAction("Export Wavefront OBJ...", self._export_obj)
        file_menu.addAction("Export Binary STL...", self._export_stl)
        file_menu.addAction("Export Stanford PLY...", self._export_ply)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit", self.close)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)

        # --- Edit Menu ---
        edit_menu = menubar.addMenu("&Edit")
        self.act_undo = edit_menu.addAction(make_icon("undo"), "Undo", self._undo)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)

        self.act_redo = edit_menu.addAction(make_icon("redo"), "Redo", self._redo)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)

        edit_menu.addSeparator()
        act_dup = edit_menu.addAction("Duplicate", self._duplicate)
        act_dup.setShortcut(QKeySequence("Ctrl+D"))

        act_del = edit_menu.addAction("Delete", self._delete_selected)
        act_del.setShortcut(QKeySequence("Delete"))

        act_sel_all = edit_menu.addAction("Select / Deselect All", self._toggle_select_all)
        act_sel_all.setShortcut(QKeySequence("A"))

        act_inv = edit_menu.addAction("Invert Selection", self._invert_selection)
        act_inv.setShortcut(QKeySequence("Ctrl+I"))

        act_loop = edit_menu.addAction("Select Edge Loop", self.inspector.action_select_edge_loop)
        act_loop.setShortcut(QKeySequence("L"))

        act_grow = edit_menu.addAction("Grow Selection", self.inspector.action_grow_selection)
        act_grow.setShortcut(QKeySequence("Ctrl++"))

        edit_menu.addSeparator()
        act_join = edit_menu.addAction("Join Selected Objects", self.inspector.action_join_objects)
        act_join.setShortcut(QKeySequence("Ctrl+J"))

        # --- Add Primitive Menu ---
        add_menu = menubar.addMenu("&Add")
        prim_icons = {
            "Cube": "cube",
            "UV Sphere": "sphere",
            "Icosphere": "sphere",
            "Cylinder": "cylinder",
            "Cone": "cone",
            "Torus": "torus",
            "Plane": "plane",
            "Pyramid": "pyramid",
        }
        for name in PRIMITIVE_BUILDERS:
            icon = make_icon(prim_icons.get(name, "cube"))
            add_menu.addAction(icon, name, lambda checked=False, n=name: self.add_primitive(n))

        # --- Mesh Modeling Menu ---
        mesh_menu = menubar.addMenu("&Mesh")
        mesh_menu.addAction(make_icon("extrude"), "Extrude Faces (E)", self.inspector.action_extrude_faces)
        mesh_menu.addAction(make_icon("inset"), "Inset Faces (I)", self.inspector.action_inset_faces)
        act_bevel = mesh_menu.addAction(make_icon("bevel"), "Bevel / Chamfer", self.inspector.action_bevel)
        act_bevel.setShortcut(QKeySequence("Ctrl+B"))
        act_lcut = mesh_menu.addAction(make_icon("loop_cut"), "Loop Cut", self.inspector.action_loop_cut)
        act_lcut.setShortcut(QKeySequence("Ctrl+R"))
        mesh_menu.addAction(make_icon("poke"), "Poke Faces", self.inspector.action_poke_faces)
        mesh_menu.addAction(make_icon("bridge"), "Bridge 2 Faces", self.inspector.action_bridge_faces)
        mesh_menu.addSeparator()
        mesh_menu.addAction(make_icon("subdivide"), "Catmull-Clark Subdivide", self.inspector.action_catmull_clark)
        mesh_menu.addAction("Simple Subdivide", self.inspector.action_subdivide_simple)
        mesh_menu.addAction(make_icon("solidify"), "Solidify Shell", self.inspector.action_solidify)
        mesh_menu.addAction(make_icon("mirror"), "Mirror & Weld", self.inspector.action_mirror)
        mesh_menu.addAction("Radial Array", self.inspector.action_radial_array)
        mesh_menu.addSeparator()
        mesh_menu.addAction(make_icon("smooth"), "Smooth Vertices", self.inspector.action_smooth)
        mesh_menu.addAction("Inflate / Shrink", self.inspector.action_inflate)
        mesh_menu.addAction("Twist (Y)", self.inspector.action_twist)
        mesh_menu.addAction("Taper (Y)", self.inspector.action_taper)
        mesh_menu.addSeparator()
        mesh_menu.addAction("Triangulate", self.inspector.action_triangulate)
        mesh_menu.addAction("Flip Normals", self.inspector.action_flip_normals)
        mesh_menu.addAction("Recalculate Outward Normals", self.inspector.action_recalculate_normals)
        mesh_menu.addAction("Weld / Merge by Distance", self.inspector.action_merge_by_distance)
        mesh_menu.addAction("Center Pivot", self.inspector.action_center_pivot)

        # --- View Menu ---
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("Frame Selected (F)", self.viewport.frame_selection)
        view_menu.addAction("Reset Camera", self._reset_camera)
        view_menu.addAction("Toggle Perspective / Ortho (5)", self._toggle_ortho)
        view_menu.addSeparator()
        for label, preset in [
            ("Front View (Numpad 1)", "front"),
            ("Back View", "back"),
            ("Right View (Numpad 3)", "right"),
            ("Left View", "left"),
            ("Top View (Numpad 7)", "top"),
            ("Bottom View", "bottom"),
            ("Isometric View", "iso"),
        ]:
            view_menu.addAction(label, lambda checked=False, p=preset: self._set_view(p))

        # --- Help Menu ---
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("Controls & Shortcuts", self._show_controls_help)

        # --------------------------------------------------------------
        # Top Studio Toolbar
        # --------------------------------------------------------------
        top_bar = QToolBar("Main Toolbar", self)
        top_bar.setMovable(False)
        top_bar.setIconSize(QSize(18, 18))
        top_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, top_bar)

        top_bar.addAction(self.act_undo)
        top_bar.addAction(self.act_redo)
        top_bar.addSeparator()

        # Mode switcher
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        for mode, label in [
            (EditMode.OBJECT, "Object"),
            (EditMode.VERTEX, "Vertex"),
            (EditMode.EDGE, "Edge"),
            (EditMode.FACE, "Face"),
        ]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.triggered.connect(lambda checked=False, m=mode: self.set_edit_mode(m))
            mode_group.addAction(act)
            top_bar.addAction(act)
            self._mode_actions[mode] = act

        top_bar.addSeparator()

        # Shading modes
        shade_group = QActionGroup(self)
        shade_group.setExclusive(True)
        for shade, label in [
            (ShadingMode.WIREFRAME, "Wire"),
            (ShadingMode.SOLID, "Solid"),
            (ShadingMode.SOLID_WIRE, "Solid+Wire"),
            (ShadingMode.SMOOTH, "Smooth"),
        ]:
            act = QAction(label, self)
            act.setCheckable(True)
            act.triggered.connect(lambda checked=False, s=shade: self.set_shading(s))
            shade_group.addAction(act)
            top_bar.addAction(act)
            self._shading_actions[shade] = act

        top_bar.addSeparator()

        # Grid & Snap toggles
        self.act_grid = QAction(make_icon("grid"), "Grid", self)
        self.act_grid.setCheckable(True)
        self.act_grid.setChecked(True)
        self.act_grid.triggered.connect(self._toggle_grid)
        top_bar.addAction(self.act_grid)

        self.act_snap = QAction("Snap", self)
        self.act_snap.setCheckable(True)
        self.act_snap.setChecked(False)
        self.act_snap.setToolTip("Snap Move (0.25) and Rotate (15°)")
        self.act_snap.triggered.connect(self._toggle_snap)
        top_bar.addAction(self.act_snap)

        # --------------------------------------------------------------
        # Left Vertical Tool Shelf (Transform + Quick Primitives)
        # --------------------------------------------------------------
        shelf = QToolBar("Tool Shelf", self)
        shelf.setObjectName("ToolShelf")
        shelf.setMovable(False)
        shelf.setIconSize(QSize(22, 22))
        shelf.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, shelf)

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        for tool, icon_key, tip in [
            (TransformTool.SELECT, "select", "Select Tool (Q)"),
            (TransformTool.MOVE, "move", "Move Tool (G / W)"),
            (TransformTool.ROTATE, "rotate", "Rotate Tool (R)"),
            (TransformTool.SCALE, "scale", "Scale Tool (S)"),
        ]:
            act = QAction(make_icon(icon_key, 22), tip, self)
            act.setToolTip(tip)
            act.setCheckable(True)
            act.triggered.connect(lambda checked=False, t=tool: self.set_tool(t))
            tool_group.addAction(act)
            shelf.addAction(act)
            self._tool_actions[tool] = act

        shelf.addSeparator()

        # Quick Add Primitives on the tool shelf
        for prim_name, icon_key in [
            ("Cube", "cube"),
            ("UV Sphere", "sphere"),
            ("Cylinder", "cylinder"),
            ("Cone", "cone"),
            ("Torus", "torus"),
            ("Pyramid", "pyramid"),
            ("Plane", "plane"),
        ]:
            act = QAction(make_icon(icon_key, 22), f"Add {prim_name}", self)
            act.setToolTip(f"Add {prim_name}")
            act.triggered.connect(lambda checked=False, n=prim_name: self.add_primitive(n))
            shelf.addAction(act)

        shelf.addSeparator()
        for icon_key, tip, slot in [
            ("extrude", "Extrude Faces (E)", self.inspector.action_extrude_faces),
            ("inset", "Inset Faces (I)", self.inspector.action_inset_faces),
            ("bevel", "Bevel / Chamfer (Ctrl+B)", self.inspector.action_bevel),
            ("loop_cut", "Loop Cut (Ctrl+R)", self.inspector.action_loop_cut),
            ("subdivide", "Catmull-Clark Subdivide", self.inspector.action_catmull_clark),
            ("mirror", "Mirror & Weld", self.inspector.action_mirror),
            ("solidify", "Solidify Shell", self.inspector.action_solidify),
            ("smooth", "Laplacian Smooth", self.inspector.action_smooth),
        ]:
            act = QAction(make_icon(icon_key, 22), tip, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            shelf.addAction(act)

    # ------------------------------------------------------------------
    # Keyboard Shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()

        # Don't intercept typing inside text inputs
        focused = self.focusWidget()
        if focused is not None and focused.inherits("QLineEdit"):
            super().keyPressEvent(event)
            return

        if key == Qt.Key.Key_Tab:
            if self.scene.edit_mode == EditMode.OBJECT:
                self.set_edit_mode(EditMode.FACE)
            else:
                self.set_edit_mode(EditMode.OBJECT)
            return

        if not (mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
            if key == Qt.Key.Key_Q:
                self.set_tool(TransformTool.SELECT)
                return
            if key in (Qt.Key.Key_G, Qt.Key.Key_W):
                self.set_tool(TransformTool.MOVE)
                return
            if key == Qt.Key.Key_R:
                self.set_tool(TransformTool.ROTATE)
                return
            if key == Qt.Key.Key_S:
                self.set_tool(TransformTool.SCALE)
                return
            if key == Qt.Key.Key_F:
                self.viewport.frame_selection()
                return
            if key == Qt.Key.Key_E and self.scene.edit_mode != EditMode.OBJECT:
                self.inspector.action_extrude_faces()
                return
            if key == Qt.Key.Key_I and self.scene.edit_mode != EditMode.OBJECT:
                self.inspector.action_inset_faces()
                return
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal) and self.scene.edit_mode != EditMode.OBJECT:
                self.inspector.action_grow_selection()
                return
            if key == Qt.Key.Key_1:
                if self.scene.edit_mode != EditMode.OBJECT:
                    self.set_edit_mode(EditMode.VERTEX)
                else:
                    self._set_view("front")
                return
            if key == Qt.Key.Key_2 and self.scene.edit_mode != EditMode.OBJECT:
                self.set_edit_mode(EditMode.EDGE)
                return
            if key == Qt.Key.Key_3:
                if self.scene.edit_mode != EditMode.OBJECT:
                    self.set_edit_mode(EditMode.FACE)
                else:
                    self._set_view("right")
                return
            if key == Qt.Key.Key_5:
                self._toggle_ortho()
                return
            if key == Qt.Key.Key_7:
                self._set_view("top")
                return

        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # State Actions & Synchronization
    # ------------------------------------------------------------------

    def _sync_all_ui(self) -> None:
        self.outliner.refresh()
        self.inspector.refresh()
        self.viewport.refresh()

        for tool, act in self._tool_actions.items():
            act.setChecked(self.scene.tool == tool)
        for mode, act in self._mode_actions.items():
            act.setChecked(self.scene.edit_mode == mode)
        for shade, act in self._shading_actions.items():
            act.setChecked(self.scene.shading == shade)

        self.act_undo.setEnabled(self.scene.can_undo)
        self.act_redo.setEnabled(self.scene.can_redo)
        self.act_grid.setChecked(self.scene.show_grid)
        self.act_snap.setChecked(self.scene.snap_enabled)

    def _show_status(self, msg: str) -> None:
        self.status_label.setText(msg)

    def set_tool(self, tool: TransformTool) -> None:
        self.scene.tool = tool
        self._sync_all_ui()
        self._show_status(f"Tool: {tool.value.capitalize()}")

    def set_edit_mode(self, mode: EditMode) -> None:
        if not self.scene.set_edit_mode(mode):
            self._show_status("Select an object before entering Edit Mode.")
            self._sync_all_ui()
            return
        self._sync_all_ui()
        self._show_status(f"Mode: {mode.value.capitalize()}")

    def set_shading(self, shade: ShadingMode) -> None:
        self.scene.shading = shade
        self._sync_all_ui()
        self._show_status(f"Shading: {shade.value.replace('_', ' ').title()}")

    def _toggle_grid(self, checked: bool) -> None:
        self.scene.show_grid = checked
        self.viewport.refresh()

    def _toggle_snap(self, checked: bool) -> None:
        self.scene.snap_enabled = checked
        self._show_status("Grid Snap ON (0.25 / 15°)" if checked else "Grid Snap OFF")

    def _toggle_ortho(self) -> None:
        self.viewport.camera.orthographic = not self.viewport.camera.orthographic
        self.viewport.refresh()
        mode = "Orthographic" if self.viewport.camera.orthographic else "Perspective"
        self._show_status(f"Projection: {mode}")

    def _set_view(self, preset: str) -> None:
        self.viewport.camera.set_preset(preset)
        self.viewport.refresh()
        self._show_status(f"View: {preset.capitalize()}")

    def _reset_camera(self) -> None:
        self.viewport.camera.reset()
        self.viewport.refresh()
        self._show_status("Camera reset")

    def add_primitive(self, name: str) -> None:
        builder = PRIMITIVE_BUILDERS.get(name)
        if builder is None:
            return
        mesh = builder()
        bmin, _bmax = mesh.compute_bounds()
        # Place primitive resting nicely on ground plane Y = 0
        y_offset = max(0.0, float(-bmin[1]))
        palette = list(MATERIAL_PRESETS.values())
        mat = palette[len(self.scene.objects) % len(palette)].clone()

        # Offset slightly on X if another object sits at origin
        x_off = 0.0
        if any(
            abs(o.position[0]) < 0.25 and abs(o.position[2]) < 0.25
            for o in self.scene.objects
        ):
            x_off = 2.2 * (((len(self.scene.objects) + 1) // 2) * (1 if len(self.scene.objects) % 2 else -1))

        obj = SceneObject(
            name=name,
            mesh=mesh,
            position=[x_off, y_offset, 0.0],
            material=mat,
        )
        self.scene.edit_mode = EditMode.OBJECT
        self.scene.add_object(obj)
        self._sync_all_ui()
        self._show_status(f"Added {obj.name}")

    def _undo(self) -> None:
        label = self.scene.undo()
        if label:
            self._sync_all_ui()
            self._show_status(f"Undo: {label}")

    def _redo(self) -> None:
        label = self.scene.redo()
        if label:
            self._sync_all_ui()
            self._show_status(f"Redo: {label}")

    def _duplicate(self) -> None:
        created = self.scene.duplicate_selected_objects()
        if created:
            self._sync_all_ui()
            self._show_status(f"Duplicated {len(created)} object(s)")

    def _delete_selected(self) -> None:
        if self.scene.edit_mode != EditMode.OBJECT:
            self.inspector.action_delete_elements()
            return
        count = self.scene.delete_selected_objects()
        if count:
            self._sync_all_ui()
            self._show_status(f"Deleted {count} object(s)")

    def _toggle_select_all(self) -> None:
        self.scene.select_all_toggle()
        self._sync_all_ui()

    def _invert_selection(self) -> None:
        self.scene.invert_selection()
        self._sync_all_ui()
        self._show_status("Inverted selection")

    # ------------------------------------------------------------------
    # File Operations (Project Save/Load & OBJ/STL/PLY Import/Export)
    # ------------------------------------------------------------------

    def _new_scene(self) -> None:
        self.scene.reset_default_scene()
        self._current_project_path = None
        self.viewport.camera.reset()
        self._sync_all_ui()
        self._show_status("Created new scene")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Py3D Project (*.p3d);;All Files (*)"
        )
        if not path:
            return
        try:
            self.scene.load_project(path)
            self._current_project_path = Path(path)
            self._sync_all_ui()
            self.viewport.frame_selection()
            self._show_status(f"Opened {self._current_project_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", str(exc))

    def _save_project(self) -> None:
        if self._current_project_path is None:
            self._save_project_as()
            return
        try:
            self.scene.save_project(self._current_project_path)
            self._show_status(f"Saved {self._current_project_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "scene.p3d", "Py3D Project (*.p3d)"
        )
        if not path:
            return
        if not path.endswith(".p3d"):
            path += ".p3d"
        self._current_project_path = Path(path)
        self._save_project()

    def _import_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import 3D Mesh", "", "3D Meshes (*.obj *.stl);;Wavefront OBJ (*.obj);;STL (*.stl)"
        )
        if not path:
            return
        p = Path(path)
        try:
            ext = p.suffix.lower()
            added = 0
            if ext == ".obj":
                groups = import_obj(p)
                for name, mesh in groups:
                    self.scene.add_object(
                        SceneObject(name=name, mesh=mesh, material=MATERIAL_PRESETS["Studio Clay"].clone())
                    )
                    added += 1
            elif ext == ".stl":
                mesh = import_stl(p)
                self.scene.add_object(
                    SceneObject(name=p.stem, mesh=mesh, material=MATERIAL_PRESETS["Studio Clay"].clone())
                )
                added = 1
            self._sync_all_ui()
            self.viewport.frame_selection()
            self._show_status(f"Imported {added} object(s) from {p.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))

    def _export_targets(self):
        selected = self.scene.selected_objects()
        return selected if selected else [o for o in self.scene.objects if o.visible]

    def _export_obj(self) -> None:
        targets = self._export_targets()
        if not targets:
            QMessageBox.information(self, "Export OBJ", "No visible objects to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Wavefront OBJ", "model.obj", "Wavefront OBJ (*.obj)")
        if not path:
            return
        if not path.endswith(".obj"):
            path += ".obj"
        export_obj(path, targets)
        self._show_status(f"Exported {len(targets)} object(s) to {Path(path).name}")

    def _export_stl(self) -> None:
        targets = self._export_targets()
        if not targets:
            QMessageBox.information(self, "Export STL", "No visible objects to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Binary STL", "model.stl", "STL Mesh (*.stl)")
        if not path:
            return
        if not path.endswith(".stl"):
            path += ".stl"
        export_stl_binary(path, targets)
        self._show_status(f"Exported {len(targets)} object(s) to {Path(path).name}")

    def _export_ply(self) -> None:
        targets = self._export_targets()
        if not targets:
            QMessageBox.information(self, "Export PLY", "No visible objects to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Stanford PLY", "model.ply", "Stanford PLY (*.ply)")
        if not path:
            return
        if not path.endswith(".ply"):
            path += ".ply"
        export_ply(path, targets)
        self._show_status(f"Exported {len(targets)} object(s) to {Path(path).name}")

    def _show_controls_help(self) -> None:
        QMessageBox.information(
            self,
            "Controls & Shortcuts",
            (
                "<b>Camera Navigation</b><br>"
                "• Orbit: Middle Mouse Drag or Alt + Left Drag<br>"
                "• Pan: Right Mouse Drag or Shift + Middle Drag<br>"
                "• Zoom: Mouse Wheel<br>"
                "• Frame Selected: <b>F</b> &nbsp;|&nbsp; Persp/Ortho: <b>5</b><br>"
                "• Views: <b>1</b> Front, <b>3</b> Right, <b>7</b> Top (or click top-right X/Y/Z triad)<br><br>"
                "<b>Selection & Transforms</b><br>"
                "• Select: Left Click (Shift+Click for multi-select)<br>"
                "• Select All: <b>A</b> &nbsp;|&nbsp; Invert: <b>Ctrl+I</b> &nbsp;|&nbsp; Grow: <b>+</b><br>"
                "• Edge Loop: <b>L</b> (Edge Mode) &nbsp;|&nbsp; Join Objects: <b>Ctrl+J</b><br>"
                "• Tools: <b>Q</b> Select, <b>G/W</b> Move, <b>R</b> Rotate, <b>S</b> Scale<br>"
                "• Duplicate: <b>Ctrl+D</b> &nbsp;|&nbsp; Delete: <b>Delete</b><br>"
                "• Undo / Redo: <b>Ctrl+Z</b> / <b>Ctrl+Y</b><br><br>"
                "<b>Polygonal Modeling</b><br>"
                "• Toggle Object / Edit Mode: <b>Tab</b> (<b>1</b> Vert, <b>2</b> Edge, <b>3</b> Face)<br>"
                "• Extrude: <b>E</b> &nbsp;|&nbsp; Inset: <b>I</b> &nbsp;|&nbsp; Bevel: <b>Ctrl+B</b> &nbsp;|&nbsp; Loop Cut: <b>Ctrl+R</b><br>"
                "• More tools in Inspector tabs: Poke, Bridge, Solidify, Mirror, Radial Array, Smooth, Inflate, Twist, Taper, Fix Normals, Weld"
            ),
        )
