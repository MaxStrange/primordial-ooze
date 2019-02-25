from setuptools import setup

with open("README.md") as f:
    doc = "\n" + f.read()

setup(
    name="primordialooze",
    version="0.3.0",
    author="Max Strange",
    author_email="maxfieldstrange@gmail.com",
    description="Super simple genetic algorithm library for Python.",
    install_requires=["numpy"],
    license="MIT",
    keywords="machine learning genetic algorithm",
    url="https://github.com/MaxStrange/primordial-ooze",
    py_modules=[],
    packages=["primordialooze"],
    python_requires="~=3.5",
    long_description=doc,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ]
)
