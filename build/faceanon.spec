# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for faceanon standalone executable."""

import os
import sys

block_cipher = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(SPECPATH)))

a = Analysis(
    [os.path.join(ROOT_DIR, 'faceanon', 'cli.py')],
    pathex=[ROOT_DIR],
    binaries=[],
    datas=[
        (os.path.join(ROOT_DIR, 'models', 'centerface.onnx'), 'models'),
    ],
    hiddenimports=['onnxruntime'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='faceanon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
