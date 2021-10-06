import asyncio
import logging
import sys
from multiprocessing import Process, Queue
from typing import Any, Tuple

from monai.deploy.exceptions import ServerError

BUFFER_SIZE = 2048

logger = logging.getLogger(__name__)


async def pipe(reader, writer):
    try:
        while not reader.at_eof():
            data = await reader.read(BUFFER_SIZE)
            writer.write(data)
    finally:
        writer.close()


async def proxy(reader, writer):
    try:
        d_reader, d_writer = await asyncio.open_unix_connection("/var/run/docker.sock")
        pipe1 = pipe(reader, d_writer)
        pipe2 = pipe(d_reader, writer)
        await asyncio.gather(pipe1, pipe2)
    finally:
        writer.close()


def get_gateway_ip(name: str = "bridge"):
    import docker

    client = docker.from_env()
    networks = client.networks.list()
    for network in networks:
        if network.name == name:
            try:
                return network.attrs["IPAM"]["Config"][0]["Gateway"]
            except KeyError as err:
                logger.info(f"networks: {networks}")
                logger.info(f"network.attrs : {network.attrs}")
                raise ServerError(f"Failed to get bridge IP: {err}") from err
    logger.warning(f"No '{name}' network found. Using 172.17.0.1")

    return "172.17.0.1"


class UnixSocketDockerServer:
    def __init__(self, host: str = "172.17.0.1", port: int = 0):
        self.host = host
        self.port = port

    def start(self, host=None, port=None):
        if host is None:
            host = self.host
        if port is None:
            port = self.port

        if sys.platform == "win32":
            # On Windows, we need to use named pipes to connect to the docker instead of '/var/run/docker.sock'.
            raise RuntimeError("Windows is not supported.")  # pragma: no cover

        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.coro = asyncio.start_server(proxy, host, port, loop=self.loop)
        self.server = self.loop.run_until_complete(self.coro)

        sock_info = self.server.sockets[0].getsockname()
        self.host, self.port = sock_info
        print(f"Docker Proxy Server is serving on {self.host}:{self.port}")

    def run_forever(self):

        try:
            self.loop.run_forever()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            self.stop()

    def stop(self):
        # Close the server
        self.server.close()
        self.loop.run_until_complete(self.server.wait_closed())
        self.loop.close()
        print("Docker Proxy Server is stopped")


def main(host=None, port=0, queue=None):
    server = UnixSocketDockerServer()
    server.start(host, port)
    if queue is not None:
        # Return the server object (host, port) to the caller
        queue.put((server.host, server.port))
    server.run_forever()


def launch_docker_proxy_server(host=None, port=0) -> Tuple[Any, str, int]:
    import signal

    def signal_handler(signum, frame):
        raise SystemExit()

    # Handle SIGTERM signal when the subprocess is terminated by terminate() method.
    signal.signal(signal.SIGTERM, signal_handler)

    queue: Queue = Queue()
    service: Process = Process(target=main, args=(host, port, queue))
    service.start()
    host, port = queue.get()

    return service, host, port


if __name__ == "__main__":

    if 1 <= len(sys.argv) <= 3:
        host = sys.argv[1] if len(sys.argv) > 1 else None
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None

        main(host, port)
    else:
        print("Usage: python3 {} [<host> [<port>]]".format(sys.argv[0]))
