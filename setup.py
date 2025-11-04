from setuptools import setup

setup(
    name="Parse",
    version="0.1",
    py_modules=["parse"],  # Match your actual filename
    entry_points={
        "console_scripts": [
            "parse=parse:main",  # This will let you type "parse" to launch
        ],
    },
)