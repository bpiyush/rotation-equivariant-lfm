"""Checks installed packages."""

packages_to_check = [
    "numpy",
    "scipy",
    "matplotlib",
    "torch",
    "torchvision",
    "tqdm",
    "kapture",
    "cv2"
]
for package in packages_to_check:
    print("Checking package: {}".format(package))
    try:
        __import__(package)
        print("Package {} is installed.".format(package))
    except ImportError:
        print(f"{package} is not installed.")
        exit(1)
    print("-" * 20)