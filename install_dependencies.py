import subprocess

def install_dependencies():
    # Upgrade pip
    subprocess.run(['pip', 'install', '--upgrade', 'pip'])

    # Install opencv-python-headless
    subprocess.run(['pip', 'install', 'opencv-python-headless'])

if __name__ == "__main__":
    install_dependencies()
