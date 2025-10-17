"""
Setup script for A2A Pipeline
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="a2a-pipeline",
    version="0.1.0",
    author="A2A Pipeline Team",
    description="Agent-to-Agent QA and Math Reasoning Pipeline",
    long_description=open("README.md", "r", encoding="utf-8").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
        "jupyter": ["jupyter", "matplotlib", "seaborn"]
    }
)

from pathlib import Path