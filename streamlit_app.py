import sys
import os
from pathlib import Path

# =======================================================================
# AI Road Quality Monitoring System - Robust Entry Point
# =======================================================================

# Get absolute path to the directory containing this file
ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "road_quality_monitor"

# Add directories to sys.path to ensure modules are found
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "3_detection"))
sys.path.insert(0, str(PROJECT_DIR / "4_dashboard"))

# Renamed 4_dashboard/app.py -> 4_dashboard/dashboard_main.py
try:
    from dashboard_main import *
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to load the dashboard: {e}")
    st.info("Check: road_quality_monitor/4_dashboard/dashboard_main.py exists")
    # Debug info
    st.write(f"Current Directory: {os.getcwd()}")
    st.write(f"Root: {ROOT}")
    st.write(f"Looking in: {PROJECT_DIR / '4_dashboard'}")
