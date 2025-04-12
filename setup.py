from setuptools import find_packages, setup
from typing import List

def get_requiremnets (file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    
    return requirements

setup(
    name="MLProject",
    version="0.0.1",
    author="Karan",
    author_email="karankarki199@gmail.com",
    packages=find_packages(),
    install_requirements= get_requiremnets('requirements.txt')
)