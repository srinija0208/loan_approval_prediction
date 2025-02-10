from setuptools import find_packages,setup

from typing import List

def get_requirements() -> List[str]:
    requirements : List[str]=[]

    return requirements



setup(
    name = "LoanApprovalPrediction",
    version = "0.0.1",
    author = "srinija",
    author_email = "srinijasomannagari@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements() # ["pymongo"]
)