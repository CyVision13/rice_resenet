import subprocess
import platform

def install_dependencies():
    # Upgrade pip
    subprocess.run(['pip', 'install', '--upgrade', 'pip'])

    # Install required pip packages
    pip_packages = ['keras', 'matplotlib', 'numpy', 'opencv-python', 'scikit-learn', 'tensorflow', 'opencv-python-headless', 'seaborn']
    subprocess.run(['pip', 'install'] + pip_packages)

    # Check if the OS is Linux
    if platform.system() == 'Linux':
        # Install required apt packages
        apt_packages = ['libgl1-mesa-glx','libglib2.0-0']
        subprocess.run(['apt-get', 'update'])
        subprocess.run(['apt-get', 'install', '-y'] + apt_packages)

if __name__ == "__main__":
    install_dependencies()
