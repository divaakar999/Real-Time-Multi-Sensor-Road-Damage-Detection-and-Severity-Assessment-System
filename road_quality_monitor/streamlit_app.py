# =======================================================================
# STREAMLIT CLOUD ENTRY POINT
# =======================================================================
# Streamlit Cloud looks for `streamlit_app.py` in the root directory by default.
# This simple wrapper imports and runs the actual dashboard code from the
# 4_dashboard folder, ensuring all paths and dependencies resolve correctly.

import sys
from pathlib import Path

# Add the 4_dashboard directory to Python's path so we can import app.py
dashboard_dir = Path(__file__).parent / "4_dashboard"
sys.path.insert(0, str(dashboard_dir))

# Import the main dashboard app
import app

# Run the dashboard's main loop
if __name__ == "__main__":
    if hasattr(app, "main"):
        app.main()
