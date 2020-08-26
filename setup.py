import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stock-forecast", 
    version="1.0.0",
    author="Yves Deutschmann",
    author_email="yves.deutschmann@gmail.com",
    description="Udacity's capstone project: stock market prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YvesDeutschmann/stock-forecast",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.3',
)