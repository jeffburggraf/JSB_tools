import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JSB_tools",
    version="0.0.1",
    author="Jeffrey Burggraf",
    author_email="jeffburggraf@gmail.com",
    description="Scientific tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jeffburggraf/JSB_tools',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)