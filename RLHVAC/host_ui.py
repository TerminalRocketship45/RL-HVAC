"""Launch the RLHVAC UI on localhost: `python host_ui.py`."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                    "--server.address", "localhost", "--server.port", "8501"])
