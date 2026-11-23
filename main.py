#!/usr/bin/env python3
"""Entry point for Py3D Modeler."""

from __future__ import annotations

import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from modeler.ui.main_window import MainWindow


def configure_opengl_format() -> None:
    """Request an OpenGL 3.3 Core Profile context with depth buffer and 4x MSAA."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> int:
    configure_opengl_format()
    app = QApplication(sys.argv)
    app.setApplicationName("Py3D Modeler")
    app.setOrganizationName("Py3D")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
