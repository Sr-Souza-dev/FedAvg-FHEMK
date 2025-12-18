from flwr.server.strategy import Strategy
from flwr.common import FitRes, Scalar, Parameters, EvaluateIns, EvaluateRes, FitIns
from typing import List, Optional, Tuple, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class FedAvg(Strategy):
    def __init__(self, 
    ):
        super().__init__()


    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """
            Initialize the server parameters.
            This method is called once at the beginning of the training process
            and can be used to set the initial model parameters.
        """
        pass

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
            Configure the next round of training.
            Selecting clients and deciding what instructions to send to these clients
        """
        pass

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
            Aggregate training results.
            Is responsible for aggregating the results returned by the clients that were selected and asked to train in configure_fit
        """
        pass

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
            Configure the next round of evaluation.
            Selecting clients and deciding what instructions to send to these clients on evaluation step
        """
        pass

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
            Aggregate evaluation results.
            Is responsible for aggregating the results returned by the clients that were selected and asked to evaluate in configure_evaluate
        """
        pass

    def evaluate(self, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
            Evaluate the current model parameters.
            Is responsible for evaluating model parameters on the server-side. (optional function)
        """
        pass
