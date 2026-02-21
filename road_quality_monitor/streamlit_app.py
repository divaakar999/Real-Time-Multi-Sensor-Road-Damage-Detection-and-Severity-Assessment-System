import sys
import os
from pathlib import Path

# =======================================================================
# AI Road Quality Monitoring System - Robust Entry Point
# =======================================================================

# Get absolute path to the directory containing this file
# Since we are in road_quality_monitor/streamlit_app.py, 
# ROOT is this folder.
ROOT = Path(__file__).resolve().parent

# Add directories to sys.path to ensure modules are found
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "3_detection"))
sys.path.insert(0, str(ROOT / "4_dashboard"))

# Renamed 4_dashboard/app.py -> 4_dashboard/dashboard_main.py
try:
    from dashboard_main import *
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to load the dashboard: {e}")
    st.info("Ensure 4_dashboard/dashboard_main.py exists and directories are correctly added to sys.path.")
    st.code(f"sys.path: {sys.path}")
