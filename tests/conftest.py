"""Shared pytest setup: import the task package and register the real-data/simulator markers."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "final" / "robotnik_arm_task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "realdata: reads the recorded Clean_253/Clean_223 datasets")
    config.addinivalue_line("markers", "sim: constructs the PyBullet simulator")
