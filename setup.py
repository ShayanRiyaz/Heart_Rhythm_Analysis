from setuptools import setup, find_packages

setup(
    name="heart_rhythm_analysis",
    version="0.1.0",
    # only search under src/ for importable packages
    package_dir={"": "src"},
    packages=find_packages(
        where="src", 
        exclude=[
            # ignore any top-level dirs you don’t want as packages
            "/archive", "archive.*",
            "notebooks", "notebooks.*",
            "build", "build.*",
            "data", "data.*",
            "trained_model", "trained_model.*",
        ]
    ),
    install_requires=[
        # your runtime deps here…
    ],
)