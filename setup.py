from setuptools import setup, find_packages

setup(
    name="PokerRL",
    version="0.1.0",
    description="A poker library for reinforcement learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Morgan Griffiths",
    author_email="rareducks101@yahoo.com",
    url="https://github.com/morgan-griffiths/PokerRL",
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here, e.g.:
        numpy>=1.19.5
        pytest>=6.2.4
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)