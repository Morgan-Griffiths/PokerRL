import os
import subprocess
import sys


def build_rust_extension():
    print("Building Rust extension...")
    rust_project_directory = os.path.join(os.getcwd(), "src", "rusteval")
    result = subprocess.run(["cargo", "build", "--release"], cwd=rust_project_directory)
    if result.returncode != 0:
        print("Error building Rust extension")
        sys.exit(result.returncode)
    # You may need to copy/move the built Rust library to a specific location here


def build_python_package():
    print("Building Python package...")
    result = subprocess.run(["poetry", "build"])
    if result.returncode != 0:
        print("Error building Python package")
        sys.exit(result.returncode)


def main():
    build_python_package()
    build_rust_extension()


if __name__ == "__main__":
    main()
