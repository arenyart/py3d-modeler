"""Scene Outliner dock panel for managing objects, visibility, locking, and renaming."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..scene import Scene


class OutlinerPanel(QWidget):
    """Displays all objects in the scene with quick actions."""

    selection_changed = Signal()
    scene_modified = Signal()
    focus_requested = Signal()

    def __init__(self, scene: Scene, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = scene
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Vis", "Lock", "Faces"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(False)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.setColumnWidth(0, 128)
        self.tree.setColumnWidth(1, 34)
        self.tree.setColumnWidth(2, 36)
        self.tree.setColumnWidth(3, 48)

        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.dup_btn = QPushButton("Duplicate")
        self.del_btn = QPushButton("Delete")
        self.dup_btn.clicked.connect(self._duplicate_selected)
        self.del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.dup_btn)
        btn_row.addWidget(self.del_btn)
        layout.addLayout(btn_row)

    def refresh(self) -> None:
        self._syncing = True
        self.tree.clear()
        sel_set = set(self.scene.selected_object_ids)
        for obj in self.scene.objects:
            item = QTreeWidgetItem(
                [
                    obj.name,
                    "👁" if obj.visible else "—",
                    "🔒" if obj.locked else "·",
                    str(obj.mesh.face_count),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, obj.obj_id)
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
            item.setTextAlignment(3, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.tree.addTopLevelItem(item)
            if obj.obj_id in sel_set:
                item.setSelected(True)
        self._syncing = False

    def _on_tree_selection(self) -> None:
        if self._syncing:
            return
        ids = [
            item.data(0, Qt.ItemDataRole.UserRole)
            for item in self.tree.selectedItems()
            if item.data(0, Qt.ItemDataRole.UserRole)
        ]
        self.scene.selected_object_ids = ids
        self.scene.active_object_id = ids[-1] if ids else None
        self.scene.clear_sub_selection()
        self.selection_changed.emit()

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        obj_id = item.data(0, Qt.ItemDataRole.UserRole)
        obj = self.scene.get_object(obj_id)
        if obj is None:
            return
        if column == 1:
            obj.visible = not obj.visible
            self.refresh()
            self.scene_modified.emit()
        elif column == 2:
            obj.locked = not obj.locked
            self.refresh()
            self.scene_modified.emit()

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        obj_id = item.data(0, Qt.ItemDataRole.UserRole)
        obj = self.scene.get_object(obj_id)
        if obj is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Object", "Name:", text=obj.name)
        if ok and new_name.strip():
            self.scene.push_undo("Rename Object")
            obj.name = new_name.strip()
            self.refresh()
            self.scene_modified.emit()

    def _show_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_focus = menu.addAction("Focus in Viewport")
        act_rename = menu.addAction("Rename...")
        act_dup = menu.addAction("Duplicate")
        menu.addSeparator()
        act_del = menu.addAction("Delete")

        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen == act_focus:
            self.focus_requested.emit()
        elif chosen == act_rename:
            self._on_item_double_clicked(item, 0)
        elif chosen == act_dup:
            self._duplicate_selected()
        elif chosen == act_del:
            self._delete_selected()

    def _duplicate_selected(self) -> None:
        if self.scene.duplicate_selected_objects():
            self.refresh()
            self.selection_changed.emit()
            self.scene_modified.emit()

    def _delete_selected(self) -> None:
        if self.scene.delete_selected_objects():
            self.refresh()
            self.selection_changed.emit()
            self.scene_modified.emit()
