import docker
import os
import shutil
import pathlib
import logging
from src import dockerfiles

logger = logging.getLogger(__name__)

def package_application(package_name: str, base_image: str, entrypoint: str, app_config: str, package_config: str):
    shutil.copy(os.path.join(os.path.dirname(dockerfiles.__file__), "template.dockerfile"), os.getcwd()) 
    pass