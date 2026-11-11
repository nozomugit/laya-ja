"""Install the tested text-only inference/training environment inside a virtualenv."""

import platform
import subprocess
import sys
from pathlib import Path

if sys.prefix == sys.base_prefix:
    sys.exit(
        "Activate a Python 3.12 virtual environment before running this installer."
    )
if sys.version_info[:2] != (3, 12):
    sys.exit("The published environment was tested with Python 3.12.")
pip = [sys.executable, "-m", "pip", "install"]
if platform.system() in {"Linux", "Windows"}:
    subprocess.run(
        pip + ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.15.0"],
        check=True,
    )
subprocess.run(
    pip + ["-r", str(Path(__file__).with_name("requirements.txt"))], check=True
)
# The Vega backend pulls in an incompatible Transformers major version.
# All four adapters use Laya's native PyTorch text engine; install its wheel without Vega.
subprocess.run(pip + ["--no-deps", "laya==0.4.31"], check=True)
