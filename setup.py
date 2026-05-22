from setuptools import setup, find_packages

setup(
    name="product-recommender",
    version="1.0.0",
    description="Hybrid Product Recommendation System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "implicit==0.7.2",
        "lightfm==1.17",
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "redis==5.0.1",
        "mlflow==2.8.1",
    ],
    python_requires=">=3.10,<3.13",
)