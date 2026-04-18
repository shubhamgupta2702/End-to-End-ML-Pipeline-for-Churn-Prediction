from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path) as f:
        requirements = f.read().splitlines()
        requirements = [req for req in requirements if req and not req.startswith("#")]
    return requirements

setup(
    name="my_ml_project",
    version="0.1.0",
    author="Shubham Gupta",
    author_email="shubhamgupta43567@gmail.com",
    description="End-to-end ML pipeline with training, evaluation, deployment, CI/CD, Monitoring using MLFlow and FastAPI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shubhamgupta2702/End-to-End-ML-Pipeline-Project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8"
        ]
    },
    entry_points={
        "console_scripts": []
    },
    include_package_data=True,
    zip_safe=False,
)