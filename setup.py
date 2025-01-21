
from setuptools import find_packages, setup
setup(
    name="open-cv",
    packages=find_packages(),
    install_requires=["torch", "torchvision",
                      "opencv-python", "ultralytics", "pyyaml", "matplotlib"],
    extras_required={"develop": []}
)
