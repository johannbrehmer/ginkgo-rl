import setuptools


setuptools.setup(
    name="ginkgo-rl",
    version="0.0.1",
    description="Reinforcement Learning algorithm for Ginkgo Jets",
    url="https://github.com/johannbrehmer/ginkgo-rl",
    author="",
    author_email="",
    license="MIT",
    packages=setuptools.find_packages(where="./"),
    package_dir={"": "./"},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
