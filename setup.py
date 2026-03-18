from setuptools import setup, find_packages

setup(
    name="arithmosdb",
    version="0.1.0",
    description="GPU-accelerated vector database",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21"],
    extras_require={
        "dev": ["pytest", "pytest-benchmark", "faiss-cpu"],
    },
)