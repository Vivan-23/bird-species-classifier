"""Launcher for Bird Species Classifier Desktop GUI."""

import os
import sys

# Ensure current directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.gui_app import launch_gui

if __name__ == "__main__":
    launch_gui()
