from setuptools import setup, find_packages

VERSION = "0.0.3"
DESCRIPTION = "Code for modularizing the training script"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    reqs = fh.read()

setup(
    name="trainer",
    version=VERSION,
    author="Vishwesh Ramanathan",
    author_email="vishwesh.ramanathan@mail.utoronto.ca",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=reqs,
    packages=find_packages(),
    keywords=["python"],
)
