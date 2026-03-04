#for installing a local package in my environment
from setuptools import setup,find_packages

setup(
    name='mcqgen',
    version='0.0.1',
    author='sanjay',
    author_emails='sanjay.s@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)