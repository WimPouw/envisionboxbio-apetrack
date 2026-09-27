from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="envisionboxbio-apetrack",
    version="0.1.0",
    author="Wim Pouw",
    author_email="w.pouw@tilburgunivesity.edu",
    description="Zero-shot multi-ape detection, tracking and pose estimation in video (OWLv2 + ByteTrack + OpenApePose).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wimpouw/envisionboxbio-apetrack",
    packages=find_packages(include=["envisionboxbio_apetrack", "envisionboxbio_apetrack.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.10",
    install_requires=required,
)
