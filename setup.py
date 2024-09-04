from setuptools import find_packages, setup

REQUIRED = [
    "jupyterlab",
]

setup(
    name="automind",
    python_requires=">=3.11.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
