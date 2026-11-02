import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "server"))
sys.path.insert(0, str(root))

# tests never touch the network or a real database
for name in ("NVIDIA_NIM_KEY_1", "NVIDIA_NIM_KEY_2", "NVIDIA_NIM_KEY_3", "DATABASE_URL", "POSTGRES_URL"):
    os.environ.pop(name, None)
os.environ["STATE_SECRET"] = "test-secret"
