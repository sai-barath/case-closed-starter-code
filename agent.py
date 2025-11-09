import subprocess
import os

if __name__ == "__main__":
    subprocess.Popen(["./steamroller"], env=os.environ.copy()).wait()
