# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec to pack Py3D Modeler into a single standalone executable.

Build with:
    pyinstaller --clean Py3DModeler.spec
Output:
    Windows: dist/Py3DModeler.exe
    Linux:   dist/Py3DModeler
    macOS:   dist/Py3DModeler.app (and dist/Py3DModeler)
"""

import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# PyOpenGL loads platform & numpy array handlers dynamically at runtime
opengl_hidden = [
    "OpenGL",
    "OpenGL.GL",
    "OpenGL.arrays",
    "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.ctypesarrays",
    "OpenGL.arrays.ctypesparameters",
    "OpenGL.arrays.ctypespointers",
    "OpenGL.arrays.lists",
    "OpenGL.arrays.nones",
    "OpenGL.arrays.numbers",
    "OpenGL.arrays.strings",
    "OpenGL.arrays.vbo",
]
if sys.platform == "win32":
    opengl_hidden += ["OpenGL.platform.win32"]
elif sys.platform == "darwin":
    opengl_hidden += ["OpenGL.platform.darwin"]
else:
    opengl_hidden += ["OpenGL.platform.glx", "OpenGL.platform.egl"]

# Exclude large unused PySide6/Qt modules to keep the single executable compact
excluded_modules = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtSql",
    "PySide6.QtNetwork",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtTest",
    "PySide6.QtXml",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtPrintSupport",
    "PySide6.QtSpatialAudio",
    "PySide6.QtTextToSpeech",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtStateMachine",
    "PySide6.QtHttpServer",
    "tkinter",
    "pytest",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=opengl_hidden + collect_submodules("modeler"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Strip unused Qt6 shared libraries that PySide6 hooks may pull in transitively
_drop_qt_libs = (
    "Qt6WebEngine", "Qt6Pdf", "Qt6Quick", "Qt6Qml", "Qt6Multimedia",
    "Qt63D", "Qt6Charts", "Qt6Graphs", "Qt6Sql", "Qt6Designer",
    "Qt6Network", "Qt6VirtualKeyboard", "Qt6ShaderTools", "Qt6SpatialAudio",
    "libQt6WebEngine", "libQt6Pdf", "libQt6Quick", "libQt6Qml", "libQt6Multimedia",
    "libQt63D", "libQt6Charts", "libQt6Graphs", "libQt6Sql", "libQt6Designer",
    "libQt6Network", "libQt6VirtualKeyboard", "libQt6ShaderTools",
)
a.binaries = [
    b for b in a.binaries
    if not any(tag in b[0] for tag in _drop_qt_libs)
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Py3DModeler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
