from setuptools import setup, find_packages

setup(
    name="phylocartoplot",
    version="0.1.0",
    description="Phylogeographic visualization: overlay phylogenetic trees on geographic raster maps",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tahiri-lab/PhyloCartoPlot",
    packages=find_packages(include=["phylocartoplot", "phylocartoplot.*"]),
    python_requires=">=3.8",
    install_requires=[
        "biopython>=1.79",
        "cartopy>=0.21",
        "rasterio>=1.3",
        "matplotlib>=3.5",
        "pandas>=1.4",
        "numpy>=1.22",
        "scikit-image>=0.19",
    ],
    extras_require={
        "dev": ["jupyter", "pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
