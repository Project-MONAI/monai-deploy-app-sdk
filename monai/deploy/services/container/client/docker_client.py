from typing import Any

from monai.deploy.services.container.core.containers import Container, DockerContainer

from .container_client import ContainerServiceClient


class DockerClient(ContainerServiceClient):
    def __init__(self, host: str = "", port: int = -1, timeout: int = 60):
        super().__init__(host, port, timeout)

        import docker

        self.client: Any = docker.from_env()

    def run(self, image, command=None, stdout=True, stderr=False, remove=False, **kwargs) -> Container:
        result = self.client.containers.run(
            image, command=command, stdout=stdout, stderr=stderr, remove=remove, **kwargs
        )
        container = DockerContainer(result)
        return container
