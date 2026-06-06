"""Cross-platform build script for faceanon standalone executable."""

import platform
import subprocess
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build():
    model_src = os.path.join(ROOT_DIR, "models", "centerface.onnx")
    if not os.path.isfile(model_src):
        print(f"Error: model file not found at {model_src}", file=sys.stderr)
        print("Run the application once to auto-download the model.", file=sys.stderr)
        sys.exit(1)

    separator = ";" if platform.system() == "Windows" else ":"
    data_spec = f"models/centerface.onnx{separator}models"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "faceanon",
        "--add-data", data_spec,
        "--hidden-import", "onnxruntime",
        "--collect-all", "onnxruntime",
        "--distpath", os.path.join(ROOT_DIR, "dist"),
        "--workpath", os.path.join(ROOT_DIR, "build", "pyinstaller_work"),
        "--specpath", os.path.join(ROOT_DIR, "build"),
        os.path.join(ROOT_DIR, "faceanon", "cli.py"),
    ]

    print(f"Building for {platform.system()} ({platform.machine()})...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode == 0:
        ext = ".exe" if platform.system() == "Windows" else ""
        output = os.path.join(ROOT_DIR, "dist", f"faceanon{ext}")
        print(f"\nBuild successful: {output}")
    else:
        print("\nBuild failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    build()
