import sys
import os
from pathlib import Path

# =======================================================================
# AI Road Quality Monitoring System - Entry Point
# =======================================================================

# Get absolute path to this file
ROOT = Path(__file__).resolve().parent

# Add directories to sys.path to ensure modules are found
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "3_detection"))
sys.path.append(str(ROOT / "4_dashboard"))

# Streamlit Cloud sometimes has issues with imports named 'app'
# We renamed 4_dashboard/app.py -> 4_dashboard/dashboard_main.py
try:
    from dashboard_main import *
except ImportError:
    # If using absolute imports
    from road_quality_monitor.main_dashboard import *
except Exception as e:
    import streamlit as st
    st.error(f"Failed to load the dashboard: {e}")
    st.info("Check that 4_dashboard/dashboard_main.py exists and dependencies are installed.")
