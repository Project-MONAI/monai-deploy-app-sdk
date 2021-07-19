import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

# Install required packages from requirements.txt file
install_requires = []
requirements_relative_path = "./requirements.txt"
package_folder = os.path.dirname(os.path.realpath(__file__))
requirements_path = package_folder + requirements_relative_path
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

# Extract version number from VERSION file
release_version = "0.0.0"
if os.path.exists('VERSION'):
    with open('VERSION') as version_file:
        release_version = version_file.read().strip()

setuptools.setup(
    name="monai-app-packager",
    author="NVIDIA Clara Deploy",
    version=release_version,
    description="Python package to build MONAI application packages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KavinKrishnan/MONAI-deploy/tree/kavink/builder",
    install_requires=install_requires,
    packages=setuptools.find_packages('.'),
    entry_points={
            'console_scripts': [
                'monaipackager = src.cli:main'
            ]
        },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    data_files=[
        ("template_dockerfile", [
         "src/dockerfiles/template.dockerfile"]),
    ],
)
