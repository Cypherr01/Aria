import os
import urllib.request
import zipfile
import subprocess
import shutil

def main():
    print("==========================================")
    print("  Starting Portable Python 3.11 Setup...")
    print("==========================================")

    workspace = r"C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria"
    os.chdir(workspace)

    # Clean up old python_312 folder if it exists
    old_folder = os.path.join(workspace, "python_312")
    if os.path.exists(old_folder):
        print("Cleaning up older Python 3.12 folder...")
        shutil.rmtree(old_folder, ignore_errors=True)

    dest_folder = os.path.join(workspace, "python_311")
    zip_path = os.path.join(workspace, "python_311.zip")

    # 1. Download and Extract Python 3.11.9
    if not os.path.exists(dest_folder):
        print("Downloading Python 3.11.9 Embeddable (Windows 64-bit)...")
        # Python 3.11 has pre-compiled Windows wheels for chroma-hnswlib!
        url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"Extracting to {dest_folder}...")
        os.makedirs(dest_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        os.remove(zip_path)
    else:
        print("[OK] Python 3.11 folder already exists.")

    # 2. Configure python311._pth to enable site-packages
    pth_file = os.path.join(dest_folder, "python311._pth")
    print(f"Configuring {pth_file}...")
    pth_content = "python311.zip\n.\nimport site\n"
    with open(pth_file, "w", encoding="ascii") as f:
        f.write(pth_content)

    # 3. Download and Install pip
    get_pip_path = os.path.join(workspace, "get-pip.py")
    if not os.path.exists(get_pip_path):
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)

    python_exe = os.path.join(dest_folder, "python.exe")
    print("Installing pip...")
    subprocess.run([python_exe, get_pip_path, "--no-warn-script-location"], check=True)
    if os.path.exists(get_pip_path):
        os.remove(get_pip_path)

    # Install packaging tools and numpy first to support packaging builds
    print("Installing setuptools, wheel, and numpy...")
    subprocess.run([python_exe, "-m", "pip", "install", "setuptools", "wheel", "numpy==1.26.4", "--no-warn-script-location"], check=True)

    # 4. Install requirements.txt
    print("Installing requirements (preferring pre-compiled binary wheels!)...")
    subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt", "--no-warn-script-location", "--prefer-binary", "--no-cache-dir"], check=True)

    # 5. Download spaCy model
    print("Downloading spaCy en_core_web_sm model...")
    subprocess.run([python_exe, "-m", "spacy", "download", "en_core_web_sm"], check=True)

    # 6. Initialize SQLite DB
    print("Initializing ARIA Database...")
    subprocess.run([python_exe, "-c", "import asyncio; from db.database import init_db; asyncio.run(init_db())"], check=True)

    print("==========================================")
    print("[OK] PORTABLE PYTHON 3.11 SETUP COMPLETE!")
    print("==========================================")

if __name__ == "__main__":
    main()
