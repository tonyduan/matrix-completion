import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="matrix-completion",
    version="0.0.2",
    author="Tony Duan",
    author_email="tonyduan@cs.stanford.edu",
    description="Python code for a few approaches at matrix completion.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonyduan/matrix-completion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
)
