import subprocess
import os
import sys

if __name__ == "__main__":
    print("==================================================")
    print("Starting ARIA Backend (FastAPI) using standard Python 3.12...")
    print("==================================================")
    
    # Path to our virtual environment Python interpreter
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = os.path.join(current_dir, "venv", "Scripts", "python.exe")
    
    if not os.path.exists(python_exe):
        print("Virtual environment python not found. Falling back to system Python.")
        python_exe = sys.executable
    else:
        print(f"Using virtual environment: {python_exe}")

    # Launch uvicorn
    try:
        subprocess.run([
            python_exe, "-m", "uvicorn", "api.main:app", 
            "--reload", "--host", "0.0.0.0", "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\nStopping backend server.")
    except Exception as e:
        print(f"\nFailed to launch backend: {e}")
