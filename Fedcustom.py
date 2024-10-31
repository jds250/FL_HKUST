from typing import Union

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

import copy
import utils


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import FedoSSL.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FedCustom(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.name_filters = ["mem_projections", "local_centroids", "local_labeled_centroids"]

        self.num_classes = 10
        self.temp_model  = models.resnet18(num_classes=10).to(device)
        self.global_model = models.resnet18(num_classes=self.num_classes).to(device)

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        num_classes = 10

        state_dict = torch.load('./FedoSSL/pretrained/simclr_cifar_10.pth.tar')
        self.global_model.load_state_dict(state_dict, strict=False)
        self.global_model = self.global_model.to(device)

        # Freeze the earlier filters
        for name, param in self.global_model.named_parameters():
            if 'linear' not in name and 'layer4' not in name:
                param.requires_grad = False
            if "centroids" in name:
                param.requires_grad = True

        ndarrays = get_parameters(self.global_model)
        return ndarrays_to_parameters(ndarrays)

    # def configure_fit(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training."""

    #     # Sample clients
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )

    #     # Create custom configs
    #     n_clients = len(clients)
    #     half_clients = n_clients // 2
    #     standard_config = {"lr": 0.001}
    #     higher_lr_config = {"lr": 0.003}
    #     fit_configurations = []
    #     for idx, client in enumerate(clients):
    #         if idx < half_clients:
    #             fit_configurations.append((client, FitIns(parameters, standard_config)))
    #         else:
    #             fit_configurations.append(
    #                 (client, FitIns(parameters, higher_lr_config))
    #             )
    #     return fit_configurations

    def add_parameters(self,w, client_model):
        for (name, server_param), client_param in zip(self.global_model.named_parameters(), client_model):
            if "centroids" not in name:
                server_param.data += client_param.data.clone() * w
            if "local_labeled_centroids" in name:
                server_param.data += client_param.data.clone() * w
                # print("Averaged layer name: ", name)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        uploaded_weights = []
        uploaded_models = []
        clients_model = []

        for i in range(len(results)):
            for j, (_, val) in enumerate(self.temp_model.state_dict().items()):
                val.copy_(torch.tensor(weights_results[i][0][j]))

            temp_model = copy.deepcopy(self.temp_model)
            uploaded_models.append(temp_model.parameters())
            clients_model.append(temp_model)

        for _,fit_res in results:
            uploaded_weights.append(1.0/len(results))

        for name, param in self.global_model.named_parameters():
            if "centroids" not in name:
                param.data = torch.zeros_like(param.data)
            if "local_labeled_centroids" in name:
                param.data = torch.zeros_like(param.data)
                # print("zeros_liked layer name: ", name)
        for w, client_model in zip(uploaded_weights, uploaded_models):
            self.add_parameters(w, client_model)

        # Run global clustering
        for client_id in range(len(results)):
            for c_name, old_param in clients_model[client_id].named_parameters():
                if "local_centroids" in c_name:
                    if client_id == 0:
                        Z1 = np.array(copy.deepcopy(old_param.data.cpu().clone()))
                    else:
                        Z1 = np.concatenate((Z1, np.array(copy.deepcopy(old_param.data.cpu().clone()))), axis=0)
        Z1 = torch.tensor(Z1, device=device).T

        self.global_model.global_clustering(Z1.to(device).T) # update self.centroids in global model
        # set labeled data feature instead of self.centroids
        self.global_model.set_labeled_feature_centroids(device=device)

        parameters_aggregated = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.global_model.state_dict().items()])
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
