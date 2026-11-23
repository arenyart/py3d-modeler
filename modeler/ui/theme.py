"""Neutral dark grey studio QSS stylesheet and programmatic vector icon generators."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: "Inter", "Segoe UI", "Ubuntu", "Cantarell", sans-serif;
    font-size: 12px;
}

QMenuBar {
    background-color: #171717;
    border-bottom: 1px solid #2e2e2e;
    padding: 2px 6px;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #303030;
    color: #ffffff;
}
QMenu {
    background-color: #232323;
    border: 1px solid #3a3a3a;
    padding: 4px;
    border-radius: 6px;
}
QMenu::item {
    padding: 6px 24px 6px 14px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #464646;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #353535;
    margin: 4px 8px;
}

QToolBar {
    background-color: #1a1a1a;
    border: none;
    border-bottom: 1px solid #2e2e2e;
    spacing: 4px;
    padding: 4px 6px;
}
QToolBar#ToolShelf {
    border-bottom: none;
    border-right: 1px solid #2e2e2e;
    padding: 8px 4px;
    spacing: 6px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 8px;
    color: #d0d0d0;
}
QToolButton:hover {
    background-color: #2e2e2e;
    border-color: #424242;
    color: #ffffff;
}
QToolButton:checked {
    background-color: #484848;
    border-color: #6a6a6a;
    color: #ffffff;
}
QToolButton:pressed {
    background-color: #3b3b3b;
}

QDockWidget {
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: 600;
}
QDockWidget::title {
    background-color: #171717;
    border-bottom: 1px solid #2e2e2e;
    border-top: 1px solid #2e2e2e;
    padding: 6px 10px;
    color: #a6a6a6;
    text-transform: uppercase;
    font-size: 11px;
}

QScrollArea {
    border: none;
    background-color: #1e1e1e;
}

QGroupBox {
    background-color: #242424;
    border: 1px solid #343434;
    border-radius: 7px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #b8b8b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    background-color: #1e1e1e;
    color: #9e9e9e;
    font-size: 11px;
}

QPushButton {
    background-color: #303030;
    border: 1px solid #444444;
    border-radius: 5px;
    padding: 5px 10px;
    color: #e6e6e6;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #3c3c3c;
    border-color: #585858;
}
QPushButton:pressed {
    background-color: #262626;
}
QPushButton#PrimaryBtn {
    background-color: #474747;
    border-color: #666666;
    color: #ffffff;
}
QPushButton#PrimaryBtn:hover {
    background-color: #545454;
    border-color: #787878;
}
QPushButton#AccentBtn {
    background-color: #504338;
    border-color: #7a614b;
    color: #f5ece4;
}
QPushButton#AccentBtn:hover {
    background-color: #615041;
    border-color: #947458;
}

QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background-color: #151515;
    border: 1px solid #383838;
    border-radius: 4px;
    padding: 2px 6px;
    color: #eaeaea;
    selection-background-color: #555555;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    width: 0px;
    border: none;
}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {
    border-color: #7d7d7d;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background-color: #232323;
    border: 1px solid #3c3c3c;
    selection-background-color: #484848;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #363636;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #7c7c7c;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #dcdcdc;
    border: 1px solid #686868;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QTreeWidget, QListWidget {
    background-color: #1a1a1a;
    border: none;
    outline: 0;
    padding: 4px;
}
QTreeWidget::item {
    padding: 5px 6px;
    border-radius: 4px;
}
QTreeWidget::item:hover {
    background-color: #2a2a2a;
}
QTreeWidget::item:selected {
    background-color: #464646;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #181818;
    color: #9a9a9a;
    padding: 4px 6px;
    border: none;
    border-bottom: 1px solid #2e2e2e;
    font-size: 11px;
}

QStatusBar {
    background-color: #161616;
    border-top: 1px solid #2d2d2d;
    color: #9e9e9e;
    font-size: 11px;
}

QScrollBar:vertical {
    background: #1a1a1a;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3a3a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #525252;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QTabWidget::pane {
    border: 1px solid #343434;
    border-radius: 5px;
    background-color: #202020;
    top: -1px;
}
QTabBar::tab {
    background-color: #181818;
    color: #969696;
    border: 1px solid #303030;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 10px;
    margin-right: 2px;
    font-size: 11px;
    font-weight: 600;
}
QTabBar::tab:hover {
    color: #e0e0e0;
    background-color: #262626;
}
QTabBar::tab:selected {
    background-color: #323232;
    color: #ffffff;
    border-color: #4d4d4d;
}
"""


def make_icon(kind: str, size: int = 20, color: str = "#dcdcdc") -> QIcon:
    """Generate a crisp anti-aliased vector icon via QPainter."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    c = QColor(color)
    pen = QPen(c, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)

    s = float(size)
    m = s * 0.16
    cx, cy = s * 0.5, s * 0.5

    if kind == "select":
        path = QPainterPath()
        path.moveTo(s * 0.28, s * 0.18)
        path.lineTo(s * 0.28, s * 0.78)
        path.lineTo(s * 0.46, s * 0.62)
        path.lineTo(s * 0.62, s * 0.84)
        path.lineTo(s * 0.72, s * 0.78)
        path.lineTo(s * 0.56, s * 0.56)
        path.lineTo(s * 0.78, s * 0.56)
        path.closeSubpath()
        p.setBrush(c)
        p.drawPath(path)

    elif kind == "move":
        p.drawLine(QPointF(cx, m), QPointF(cx, s - m))
        p.drawLine(QPointF(m, cy), QPointF(s - m, cy))
        a = s * 0.13
        p.drawLine(QPointF(cx, m), QPointF(cx - a, m + a))
        p.drawLine(QPointF(cx, m), QPointF(cx + a, m + a))
        p.drawLine(QPointF(cx, s - m), QPointF(cx - a, s - m - a))
        p.drawLine(QPointF(cx, s - m), QPointF(cx + a, s - m - a))
        p.drawLine(QPointF(m, cy), QPointF(m + a, cy - a))
        p.drawLine(QPointF(m, cy), QPointF(m + a, cy + a))
        p.drawLine(QPointF(s - m, cy), QPointF(s - m - a, cy - a))
        p.drawLine(QPointF(s - m, cy), QPointF(s - m - a, cy + a))

    elif kind == "rotate":
        rect = QRectF(m, m, s - 2 * m, s - 2 * m)
        p.drawArc(rect, 35 * 16, 290 * 16)
        p.drawLine(QPointF(s - m - 2, m + 1), QPointF(s - m - 2, m + 5.5))
        p.drawLine(QPointF(s - m - 2, m + 5.5), QPointF(s - m - 6.5, m + 4.5))

    elif kind == "scale":
        p.drawRect(QRectF(m + 2, s - m - 6, 4.5, 4.5))
        p.drawRect(QRectF(s - m - 6.5, m + 1.5, 5, 5))
        p.drawLine(QPointF(m + 6.5, s - m - 6), QPointF(s - m - 6.5, m + 6.5))

    elif kind == "cube":
        top = QPointF(cx, m)
        r_top = QPointF(s - m, s * 0.34)
        l_top = QPointF(m, s * 0.34)
        mid = QPointF(cx, s * 0.52)
        bot = QPointF(cx, s - m)
        r_bot = QPointF(s - m, s * 0.72)
        l_bot = QPointF(m, s * 0.72)
        p.drawPolygon([top, r_top, mid, l_top])
        p.drawPolygon([l_top, mid, bot, l_bot])
        p.drawPolygon([mid, r_top, r_bot, bot])

    elif kind == "sphere":
        r = s * 0.34
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.drawEllipse(QPointF(cx, cy), r, r * 0.40)
        p.drawEllipse(QPointF(cx, cy), r * 0.42, r)

    elif kind == "cylinder":
        rx, ry = s * 0.28, s * 0.11
        p.drawEllipse(QPointF(cx, m + ry + 1), rx, ry)
        p.drawLine(QPointF(cx - rx, m + ry + 1), QPointF(cx - rx, s - m - ry - 1))
        p.drawLine(QPointF(cx + rx, m + ry + 1), QPointF(cx + rx, s - m - ry - 1))
        p.drawArc(QRectF(cx - rx, s - m - 2 * ry - 1, 2 * rx, 2 * ry), 180 * 16, 180 * 16)

    elif kind == "cone":
        rx, ry = s * 0.30, s * 0.11
        p.drawLine(QPointF(cx, m), QPointF(cx - rx, s - m - ry))
        p.drawLine(QPointF(cx, m), QPointF(cx + rx, s - m - ry))
        p.drawEllipse(QPointF(cx, s - m - ry), rx, ry)

    elif kind == "torus":
        p.drawEllipse(QPointF(cx, cy), s * 0.35, s * 0.24)
        p.drawEllipse(QPointF(cx, cy), s * 0.14, s * 0.08)

    elif kind == "plane":
        p.drawPolygon([
            QPointF(s * 0.25, s * 0.32),
            QPointF(s * 0.86, s * 0.32),
            QPointF(s * 0.75, s * 0.72),
            QPointF(s * 0.14, s * 0.72),
        ])

    elif kind == "pyramid":
        apex = QPointF(cx, m)
        p.drawLine(apex, QPointF(m, s * 0.72))
        p.drawLine(apex, QPointF(cx + 1, s - m))
        p.drawLine(apex, QPointF(s - m, s * 0.68))
        p.drawLine(QPointF(m, s * 0.72), QPointF(cx + 1, s - m))
        p.drawLine(QPointF(cx + 1, s - m), QPointF(s - m, s * 0.68))

    elif kind == "extrude":
        p.drawRect(QRectF(m + 1, cy, s - 2 * m - 2, s * 0.32))
        p.setPen(QPen(QColor("#e59d4d"), 1.8))
        p.drawRect(QRectF(m + 4, m + 1, s - 2 * m - 8, s * 0.26))
        p.drawLine(QPointF(cx, cy), QPointF(cx, m + 3))

    elif kind == "inset":
        p.drawRect(QRectF(m, m, s - 2 * m, s - 2 * m))
        p.setPen(QPen(QColor("#e59d4d"), 1.6))
        p.drawRect(QRectF(m + 4, m + 4, s - 2 * m - 8, s - 2 * m - 8))

    elif kind == "subdivide":
        r = QRectF(m, m, s - 2 * m, s - 2 * m)
        p.drawRect(r)
        p.drawLine(QPointF(cx, m), QPointF(cx, s - m))
        p.drawLine(QPointF(m, cy), QPointF(s - m, cy))

    elif kind == "grid":
        step = (s - 2 * m) / 3.0
        for i in range(4):
            off = m + i * step
            p.drawLine(QPointF(off, m), QPointF(off, s - m))
            p.drawLine(QPointF(m, off), QPointF(s - m, off))

    elif kind == "undo":
        p.drawArc(QRectF(m + 2, m + 3, s - 2 * m - 2, s - 2 * m - 2), 0, 160 * 16)
        p.drawLine(QPointF(m + 2, cy - 1), QPointF(m + 2, m + 3))
        p.drawLine(QPointF(m + 2, cy - 1), QPointF(m + 6.5, cy - 1))

    elif kind == "redo":
        p.drawArc(QRectF(m, m + 3, s - 2 * m - 2, s - 2 * m - 2), 20 * 16, 160 * 16)
        p.drawLine(QPointF(s - m - 2, cy - 1), QPointF(s - m - 2, m + 3))
        p.drawLine(QPointF(s - m - 2, cy - 1), QPointF(s - m - 6.5, cy - 1))

    elif kind == "bevel":
        p.drawPolygon([
            QPointF(m + 4, m),
            QPointF(s - m - 4, m),
            QPointF(s - m, m + 4),
            QPointF(s - m, s - m - 4),
            QPointF(s - m - 4, s - m),
            QPointF(m + 4, s - m),
            QPointF(m, s - m - 4),
            QPointF(m, m + 4),
        ])
        p.setPen(QPen(QColor("#e59d4d"), 1.8))
        p.drawLine(QPointF(s - m - 4, m), QPointF(s - m, m + 4))
        p.drawLine(QPointF(m, m + 4), QPointF(m + 4, m))

    elif kind == "loop_cut":
        p.drawRect(QRectF(m, m, s - 2 * m, s - 2 * m))
        p.setPen(QPen(QColor("#e59d4d"), 2.0))
        p.drawLine(QPointF(cx, m - 1), QPointF(cx, s - m + 1))

    elif kind == "poke":
        p.drawRect(QRectF(m, m, s - 2 * m, s - 2 * m))
        p.setPen(QPen(QColor("#e59d4d"), 1.4))
        p.drawLine(QPointF(m, m), QPointF(s - m, s - m))
        p.drawLine(QPointF(s - m, m), QPointF(m, s - m))

    elif kind == "bridge":
        p.drawRect(QRectF(m, m + 2, s * 0.22, s - 2 * m - 4))
        p.drawRect(QRectF(s - m - s * 0.22, m + 2, s * 0.22, s - 2 * m - 4))
        p.setPen(QPen(QColor("#e59d4d"), 1.8))
        p.drawLine(QPointF(m + s * 0.22, m + 4), QPointF(s - m - s * 0.22, m + 4))
        p.drawLine(QPointF(m + s * 0.22, s - m - 4), QPointF(s - m - s * 0.22, s - m - 4))

    elif kind == "mirror":
        p.drawPolygon([QPointF(m, m + 2), QPointF(cx - 2.5, cy), QPointF(m, s - m - 2)])
        p.setPen(QPen(QColor("#a8a8a8"), 1.5, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(cx, m - 1), QPointF(cx, s - m + 1))
        p.setPen(QPen(QColor("#e59d4d"), 1.6))
        p.drawPolygon([QPointF(s - m, m + 2), QPointF(cx + 2.5, cy), QPointF(s - m, s - m - 2)])

    elif kind == "solidify":
        p.drawRect(QRectF(m, m + 2, s - 2 * m - 3, s - 2 * m - 3))
        p.setPen(QPen(QColor("#e59d4d"), 1.6))
        p.drawRect(QRectF(m + 3, m - 1, s - 2 * m - 3, s - 2 * m - 3))

    elif kind == "smooth":
        path = QPainterPath()
        path.moveTo(m, cy + 3)
        path.cubicTo(s * 0.35, m, s * 0.65, s - m, s - m, cy - 3)
        p.setPen(QPen(QColor("#d4d4d4"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(path)

    p.end()
    return QIcon(pm)
