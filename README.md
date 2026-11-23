# Py3D Modeler (Python + PySide6)

A desktop polygonal 3D modeling application built with **Python**, **PySide6 (Qt 6)**, **OpenGL 3.3 Core Profile**, and **NumPy**.

## Features

- **Real-Time OpenGL 3.3 Core Viewport**
  - Three-point studio lighting with roughness/metallic shading, Fresnel rim highlights, ACES filmic tone mapping, and gradient studio backdrop
  - Shading modes: `Wire`, `Solid`, `Solid+Wire`, and `Smooth`
  - Turntable Orbit, Pan, Zoom, Perspective/Orthographic projection toggle, View presets (`Front`, `Right`, `Top`, `Iso`), and interactive top-right X/Y/Z orientation triad
  - Interactive 3D Transform Gizmo for **Move**, **Rotate**, and **Scale** (axis-constrained or free view-plane drag, with optional grid/angle snapping)
- **Polygonal Modeling & Edit Mode**
  - Switch between **Object Mode** and **Vertex / Edge / Face Edit Modes** (`Tab`, `1/2/3`)
  - Raycasted Object, Face, Edge, and Vertex picking (`Shift+Click` multi-select), plus **Edge Loop Select (`L`)**, **Grow Selection (`+`)**, and **Invert Selection (`Ctrl+I`)**
  - **Edit Tools**: Extrude Faces, Inset Faces, **Bevel / Chamfer**, **Loop Cut** (multi-cut quad ring slicer), **Poke Faces** (with normal offset), **Bridge 2 Faces**, and Delete Elements
  - **Modify Tools**: Catmull-Clark Subdivision, Simple Subdivision, **Solidify Shell**, **Mirror & Weld** (X/Y/Z), **Radial Array**, and **Join Selected Objects** (`Ctrl+J`)
  - **Deform & Cleanup Tools**: **Laplacian Smooth**, **Inflate / Shrink**, **Twist**, **Taper**, Triangulate, Flip Normals, **Recalculate Outward Normals**, **Weld / Merge by Distance**, and Center Pivot
- **Procedural 3D Primitives**
  - Cube, UV Sphere, Icosphere, Cylinder, Cone, Torus, Pyramid, and Subdivided Plane
- **Scene Management & Materials**
  - Scene Outliner (visibility, lock, rename, duplicate, delete, focus)
  - Inspector (Position/Rotation/Scale fields, custom color picker, roughness/metallic sliders, 8 studio material presets)
  - Multi-step **Undo / Redo** (`Ctrl+Z` / `Ctrl+Y`)
- **Project & 3D Mesh I/O**
  - Save / Load native `.p3d` scene projects
  - Import & Export **Wavefront `.obj`** and **Binary/ASCII `.stl`**, plus **Stanford `.ply`** export

## Quick Start

```bash
cd /home/antigravity/py3d_modeler
source .venv/bin/activate
python main.py
```

Or install into your own environment:

```bash
pip install -r requirements.txt
python main.py
```

## Packaging into a Single `.exe` (Windows) or Binary (Linux/macOS)

PyInstaller packages Python, PySide6, PyOpenGL, NumPy, and `modeler/` into a single standalone executable (trimmed of unused Qt modules so it stays compact).

**On Windows** — double-click `build_exe.bat`, or run:

```bat
build_exe.bat
```

Output: `dist\Py3DModeler.exe`

**On Linux / macOS**:

```bash
./build_binary.sh
```

Output: `dist/Py3DModeler`

Or run PyInstaller manually on any OS:

```bash
pip install pyinstaller
pyinstaller --clean --noconfirm Py3DModeler.spec
```

## Controls & Shortcuts

| Action | Control / Shortcut |
| --- | --- |
| Orbit Camera | Middle Mouse Drag or `Alt + Left Drag` |
| Pan Camera | Right Mouse Drag or `Shift + Middle Drag` |
| Zoom Camera | Mouse Wheel |
| Select / Multi-Select | Left Click / `Shift + Left Click` |
| Select All / Invert / Grow | `A` / `Ctrl+I` / `+` |
| Select Edge Loop (Edge Mode) | `L` |
| Transform Tools | `Q` Select · `G`/`W` Move · `R` Rotate · `S` Scale |
| Toggle Object / Edit Mode | `Tab` |
| Edit Sub-Modes | `1` Vertex · `2` Edge · `3` Face |
| Extrude / Inset Faces | `E` / `I` |
| Bevel / Loop Cut | `Ctrl+B` / `Ctrl+R` |
| Join Selected Objects | `Ctrl+J` |
| Frame Selected | `F` |
| Perspective / Ortho | `5` |
| View Presets (Object Mode) | `1` Front · `3` Right · `7` Top |
| Duplicate / Delete | `Ctrl+D` / `Delete` |
| Undo / Redo | `Ctrl+Z` / `Ctrl+Y` |
# py3d-modeler
