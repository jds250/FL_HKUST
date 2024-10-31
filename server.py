import argparse
import logging

import flwr as fl
from flwr.server.strategy import FedAvg,FedProx
from Fedcustom import FedCustom

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Parse command line arguments
parser = argparse.ArgumentParser(description="Flower Server")
parser.add_argument(
    "--number_of_rounds",
    type=int,
    default=50,
    help="Number of FL rounds (default: 100)",
)
args = parser.parse_args()


def start_fl_server(strategy, rounds):
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_fl_server(strategy=FedCustom(),
                     rounds=args.number_of_rounds)

# FedProx(proximal_mu = 2.0,fraction_fit = 0.4)