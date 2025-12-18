
from logging import WARNING
from typing import Callable, Optional, Union
from utils.files import register_logs

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import (
    aggregate,
    aggregate_inplace,
    weighted_loss_avg,
)
from flwr.common.logger import log

from fl_simulation.ckks_instance import ckks
from ckks.cryptogram.main import Cryptogram
from ckks.polynomials.main import Polynomials

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

clientId = 0
createdNodesId = {}

def aggregate_ndarrays(array: list[list[Cryptogram]]) -> list[Cryptogram]:
    """
    Aggregate a list of arrays using weighted average.
    """
    # Get the number of clients
    num_clients = len(array)
    # Get the number of weights
    num_weights = len(array[0])
    # Create an array to store the aggregated weights
    aggregated_weights = []
    for i in range(num_weights):
        # Create an array to store the aggregated weights for each client
        aggregated_weights.append(Cryptogram(c0=Polynomials(coefficients=[]), c1=Polynomials(coefficients=[]), q=ckks.params.qs))
        for j in range(num_clients):
            # Add the weights for each client
            aggregated_weights[i] += array[j][i]
    
    return aggregated_weights

class FedAvg(Strategy):
    def __init__(
        self,
        *,
        is_flattened: bool = False,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.is_flattened = is_flattened

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Parameters and config
        ckks.gen_new_fixed_a()
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        config = {
            "server_round": server_round,
            "is_flattened": self.is_flattened,
            "clients_qtd": len(clients),
        }

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []
        
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Parameters and config
        config = {
            "server_round": server_round,
            "is_flattened": self.is_flattened,
            "clients_qtd": len(clients),
        }

        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results or (not self.accept_failures and failures):
            return None, {}
        
        register_logs(file_name="server", title=f"\n ------------- Round {server_round} -------------- \n", value="")
        register_logs(file_name="server", title="\n Results:", value=repr(results))

        if self.is_flattened:
            sk = Polynomials(coefficients=[])
            #  sk = ckks.load_key(prefix="server")
            register_logs(file_name="server", title="\n server Keys:", value=repr(sk.coefficients))

            # Convert to an array of cryptograms
            weights_cypher = []
            total_examples = 0
            for client_idx, (client, res) in enumerate(results):
                model_enc = ckks.construct_cryptograms(parameters_to_ndarrays(res.parameters))
                
                # Weighted average: multiply by num_examples
                num_examples = res.num_examples
                total_examples += num_examples
                
                # Multiply each cryptogram by num_examples
                model_enc = [c * num_examples for c in model_enc]

                weights_cypher.append(model_enc)

                sk_cl = ckks.load_key(prefix=str(client.node_id))
                sk += sk_cl
                register_logs(file_name="server", title=f"\n Client Model ({client.node_id}):", value=repr(
                    ckks.decrypt_batch(sk=sk_cl, ciphertexts=model_enc)
                ))
                register_logs(file_name="server", title=f"\n Client A ({client.node_id}):", value=repr(
                    model_enc[0].c1.coefficients
                ))

            register_logs(file_name="server", title="\n Models encripted:", value=repr(weights_cypher))

            # Agregate all clients ciphertexts
            aggregated_ndarrays = aggregate_ndarrays(weights_cypher)
            register_logs(file_name="server", title=f"\n Server agg A:", value=repr(
                aggregated_ndarrays[0].c1.coefficients
            ))
            register_logs(file_name="server", title="\n Model encripted:", value=repr(aggregated_ndarrays))

            aggregated_ndarrays = ckks.decrypt_batch(sk=sk, ciphertexts=aggregated_ndarrays)
            
            # Divide by the total number of examples to get the weighted average
            if total_examples > 0:
                aggregated_ndarrays = aggregated_ndarrays / total_examples

            register_logs(file_name="server", title="\n Model decripted:", value=repr(aggregated_ndarrays))
            parameters_aggregated = ndarrays_to_parameters([aggregated_ndarrays])

        else:
            weights_results = [
                (parameters_to_ndarrays(res.parameters), res.num_examples)
                for client_idx, (_, res) in enumerate(results)
            ]
            aggregated_ndarrays = aggregate(weights_results)                # Realiza a agregação dos parâmetros por média ponderada
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        register_logs(file_name="server", title="\n Model paramenters:", value=repr(parameters_aggregated))
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated