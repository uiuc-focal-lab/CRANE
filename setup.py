import setuptools
import os

# Get the directory where setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))
# The package is in the parent directory (src/crane)
package_dir = os.path.dirname(setup_dir)

setuptools.setup(
    name="crane",
    version="0.1",
    description="Iterative generation package for CRANE",
    packages=["crane"],
    package_dir={"crane": "src/crane"},
    include_package_data=True,
    install_requires=open("src/crane/requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

