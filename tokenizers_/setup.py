from setuptools import setup, find_packages

setup(
    name='mytokenizers',
    version='1.0.0',
    description='',
    author='Liang Xian Bing',
    author_email='',
    url='',
    install_requires=[
        'numpy',
    ],
    # entry_points={
    #     'console_scripts': [
    #         ''
    #     ]
    # },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)