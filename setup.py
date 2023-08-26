# This file helps to convert my entire machine learning project as a package which can be used later on in different projects and can be deployed on PYpi
from setuptools import find_packages,setup
from typing import List
Hyphen_e_dot = "-e ."
def get_requirements(file_path)->List[str]:
    '''
    this function is going to return the list of requirements

    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements =[req.replace("\n","") for req in requirements]

        if Hyphen_e_dot in requirements:
            requirements.remove(Hyphen_e_dot)

setup(
    name = "mlproject",
    version = "0.0.1",
    author  = 'hitesh',
    author_email='hitesh.ms24@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirement.txt')

)