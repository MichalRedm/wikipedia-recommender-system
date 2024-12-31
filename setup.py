from setuptools import setup, find_packages

setup(
    name="wikipedia-recommender-system",
    version="0.1.0",
    author="Jędrzej Miczke, Michał Redmer",
    description="A recommender system for Wikipedia articles based on TF-IDF similarity.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MichalRedm/wikipedia-recommender-system",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    include_package_data=True,
    install_requires=[
        "beautifulsoup4==4.12.3",
        "bs4==0.0.2",
        "certifi==2024.8.30",
        "charset-normalizer==3.4.0",
        "click==8.1.7",
        "colorama==0.4.6",
        "exceptiongroup==1.2.2",
        "idna==3.10",
        "iniconfig==2.0.0",
        "joblib==1.4.2",
        "nltk==3.9.1",
        "numpy==1.26.4",
        "packaging==24.2",
        "pandas==2.2.3",
        "pluggy==1.5.0",
        "pytest==8.3.3",
        "python-dateutil==2.9.0.post0",
        "pytz==2024.2",
        "regex==2024.11.6",
        "requests==2.32.3",
        "scikit-learn==1.5.2",
        "scipy==1.11.3",
        "six==1.16.0",
        "soupsieve==2.6",
        "threadpoolctl==3.5.0",
        "tomli==2.2.1",
        "tqdm==4.67.1",
        "tzdata==2024.2",
        "urllib3==2.2.3",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "create-recommender=scripts.create_recommender:main",
            "recommend=scripts.recommend:main",
        ],
    },
    keywords="Wikipedia recommender TF-IDF scikit-learn",
)
