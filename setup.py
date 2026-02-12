"""
Setup configuration for PII/PHI Classifier package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pii-phi-classifier",
    version="1.0.0",
    author="Your Organization",
    author_email="data-governance@example.com",
    description="Automated PII/PHI classification system using Vertex AI and PaLM 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/pii-phi-classifier",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/pii-phi-classifier/issues",
        "Documentation": "https://github.com/your-org/pii-phi-classifier/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Data Processing :: Privacy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "pylint>=3.0.0",
            "mypy>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pii-discover=discovery.schema_discovery:main",
            "pii-profile=profiling.field_profiler:main",
            "pii-classify=classification.palm2_classifier:main",
            "pii-train=training.training_data:main",
            "pii-finetune=training.fine_tuning:main",
            "pii-mask=masking.dlp_masking:main",
        ],
    },
)
