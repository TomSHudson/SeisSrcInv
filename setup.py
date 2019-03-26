import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SeisSrcInv",
    version="0.0.1",
    author="Tom Hudson",
    author_email="tsh37@cam.ac.uk",
    description="A full waveform seismic source mechanism inversion package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomSHudson/SeisSrcInv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'os', 'sys', 'pickle', 'random', 'math', 'multiprocessing', 'matplotlib', 'obspy', 'glob'],
)
