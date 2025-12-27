from setuptools import setup, find_packages

setup(
    name="muon_doom",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.26,<2.0",
        "opencv-python>=4.8.0",
        "vizdoom>=1.2.0",
        "pufferlib>=3.0.0",
        "torch>=2.0.0",
        "heavyball>=2.1.0",
    ],
    python_requires=">=3.9",
)
