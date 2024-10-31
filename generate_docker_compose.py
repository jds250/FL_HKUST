import argparse
import random

parser = argparse.ArgumentParser(description="Generated Docker Compose")
parser.add_argument(
    "--total_clients", type=int, default=2, help="Total clients to spawn (default: 2)"
)
parser.add_argument(
    "--num_rounds", type=int, default=40, help="Number of FL rounds (default: 100)"
)
parser.add_argument(
    "--data_percentage",
    type=float,
    default=0.6,
    help="Portion of client data to use (default: 0.6)",
)
parser.add_argument(
    "--random", action="store_true", help="Randomize client configurations"
)

def create_docker_compose(args):
    # cpus is used to set the number of CPUs available to the container as a fraction of the total number of CPUs on the host machine.
    # mem_limit is used to set the memory limit for the container.
    client_configs = [
        {"mem_limit": "1g", "batch_size": 32, "cpus": 4, "learning_rate": 0.001},
        {"mem_limit": "1g", "batch_size": 256, "cpus": 1, "learning_rate": 0.05},
        {"mem_limit": "1g", "batch_size": 64, "cpus": 3, "learning_rate": 0.02},
        {"mem_limit": "1g", "batch_size": 128, "cpus": 2.5, "learning_rate": 0.09},
        # Add or modify the configurations depending on your host machine
    ]

    docker_compose_content = f"""
version: '3'
services:
  server:
    container_name: server
    build:
      context: .
      dockerfile: Dockerfile
    command: python server.py --number_of_rounds={args.num_rounds}
    environment:
      FLASK_RUN_PORT: 6000
      DOCKER_HOST_IP: host.docker.internal
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock      
    ports:
      - "6000:6000"
      - "8265:8265"
      - "8000:8000"
"""
    # Add client services
    for i in range(0, args.total_clients):
        if args.random:
            config = random.choice(client_configs)
        else:
            config = client_configs[(i - 1) % len(client_configs)]
        docker_compose_content += f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: Dockerfile
    command: python client.py --server_address=server:8080   --client_id={i} --total_clients={args.total_clients} --batch_size={config["batch_size"]}
    deploy:
      resources:
        limits:
          cpus: "{(config['cpus'])}"
          memory: "{config['mem_limit']}"
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "{6000 + i}:{6000 + i}"
    depends_on:
      - server
    environment:
      FLASK_RUN_PORT: {6000 + i}
      container_name: client{i}
      DOCKER_HOST_IP: host.docker.internal
"""

    # docker_compose_content += "volumes:\n"

    with open("docker-compose.yml", "w") as file:
        file.write(docker_compose_content)


if __name__ == "__main__":
    args = parser.parse_args()
    create_docker_compose(args)
