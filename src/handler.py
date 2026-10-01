"""
RunPod Serverless Handler - Entry Point
=======================================
This is the canonical entry point for RunPod Serverless GitHub integration.
RunPod's scanner looks for runpod.serverless.start() in src/handler.py.
"""

import runpod
import sys
import os

# Add the workspace root to the Python path so we can import the app package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the full handler logic from the root handler module
from handler import handler


# --- RunPod Serverless Entry Point ---
runpod.serverless.start({"handler": handler})
