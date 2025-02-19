from __future__ import annotations
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives import BaseEstimator, BaseEstimatorV1, Estimator, EstimatorResult
from qiskit.quantum_info import Statevector,SparsePauliOp,DensityMatrix,Pauli
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit_machine_learning.optimizers import Optimizer, OptimizerResult, Minimizer
import qiskit_machine_learning.optionals as _optionals
from qiskit_machine_learning.algorithms.trainable_model import TrainableModel
from typing import Any

from qiskit_machine_learning.algorithms.objective_functions import BinaryObjectiveFunction

from qiskit_algorithms.utils.validation import validate_min

from qiskit_machine_learning.algorithms.objective_functions import ObjectiveFunction
from typing import Callable
if _optionals.HAS_SPARSE:
    # pylint: disable=import-error
    from sparse import SparseArray
else:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass

import ray

@ray.remote
def process_data_ray(data_i, ansatz_bound, operator):
    input_data_i = DensityMatrix(data_i)
    output_data_i = input_data_i.evolve(ansatz_bound)
    return output_data_i.expectation_value(operator)

def parallel_process_ray(input_data, ansatz_bound, operator):
    # Distribute tasks to Ray workers
    futures = [process_data_ray.remote(data_i, ansatz_bound, operator) for data_i in input_data]
    # Collect results
    results = ray.get(futures)
    return results
        
class MixedState_EstimatorQNN(EstimatorQNN):
    def __init__(self,ansatz: QuantumCircuit,**kwargs):
        #super(MixedState_EstimatorQNN, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self._ansatz = ansatz.copy()

    def forward(
        self,
        input_data: float | list[float] | np.ndarray | None,
        weights: float | list[float] | np.ndarray | None,
    ) -> np.ndarray | SparseArray:
        """Forward pass of the network.

        Args:
            input_data: input data of the shape (num_inputs). In case of a single scalar input it is
                directly cast to and interpreted like a one-element array.
            weights: trainable weights of the shape (num_weights). In case of a single scalar weight
                it is directly cast to and interpreted like a one-element array.
        Returns:
            The result of the neural network of the shape (output_shape).
        """

        #input_, shape = self._validate_input(input_data)
        #weights_ = self._validate_weights(weights)
        
        output_data = self._forward(input_data, weights)
        assert(not any(output_data)<-1 and not any(output_data)>1) ## output [-1,1]
        #output_data= (output_data+1)/2
        output_data= 1/(1 + np.exp(-10*output_data)) ##output [0,1] with sigmoid
        #return self._validate_forward_output(output_data, shape)

        return output_data

    def _forward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> np.ndarray | None:
        """Forward pass of the neural network."""
        n_qubit=self._circuit.num_qubits

        if isinstance(self.estimator, BaseEstimatorV1):
            # Here, it is rather just the ansatz. The input is handled directly as the input Density Matrix.
            ansatz_bound=self._ansatz.assign_parameters(weights)
            assert(len(self._observables)==1, "Only one observable is allowed but got multiple")
            operator=self._observables[0]

            #results = parallel_process_ray(input_data, ansatz_bound, operator)
            results=[]
            for data_i in input_data: ### how to paralellize this?
                input_data_i=DensityMatrix(data_i)
                assert(2**ansatz_bound.num_qubits==len(input_data_i.data[0]))
                output_data_i=input_data_i.evolve(ansatz_bound)
                operator_expectation_value=output_data_i.expectation_value(operator)

                results.append(operator_expectation_value)

        elif isinstance(self.estimator, BaseEstimatorV2):
            raise QiskitMachineLearningError(
                "The accepted estimators is only BaseEstimatorV1; got BaseEstimatorV2 instead. "
            )
        else:
            raise QiskitMachineLearningError(
                "The accepted estimators are BaseEstimatorV1 and BaseEstimatorV2; got "
                + f"{type(self.estimator)} instead. Note that BaseEstimatorV1 is deprecated in"
                + "Qiskit and removed in Qiskit IBM Runtime."
            )
        
        return np.real_if_close(results)
    


class MixedState_BinaryObjectiveFunction(BinaryObjectiveFunction):
    def objective(self, weights: np.ndarray) -> float:
        # predict is of shape (N, 1), where N is a number of samples
        predict = self._neural_network_forward(weights)
        target = np.array(self._y).reshape(predict.shape)
        return float(np.sum(self._loss(predict, target)) / self._num_samples)

class MixedState_NNClassifier(NeuralNetworkClassifier):
    def __init__(self,**kwargs):
        super(MixedState_NNClassifier, self).__init__(**kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X, _ = self._validate_input(X)

        if self._neural_network.output_shape == (1,):
            # Binary classification
            raw_output = self._neural_network.forward(X, self._fit_result.x)
            raw_output=raw_output*2-1 #Change the neural network output from (0,1) to (-1,1) to use np.sign for prediction
            predict = (np.sign(raw_output)+1)*1./2

        else:
            # Multi-class classification
            forward = self._neural_network.forward(X, self._fit_result.x)
            predict_ = np.argmax(forward, axis=1)

            if self._one_hot:
                # Convert class indices to one-hot encoded format
                predict = np.zeros(forward.shape)
                for i, v in enumerate(predict_):
                    predict[i, v] = 1
            else:
                predict = predict_

        return self._validate_output(predict)

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X, _ = self._validate_input(X)

        if self._neural_network.output_shape == (1,):
            # Binary classification
            raw_output = self._neural_network.forward(X, self._fit_result.x)
            #raw_output=raw_output*2-1 #Change the neural network output from (0,1) to (-1,1) for binary classification, use in combination with MixedState_EstimatorQNN and 'absolute error'
            
        return raw_output

    def _fit_internal(self, X: np.ndarray,y: np.ndarray) -> OptimizerResult:
        X, y = self._validate_input(X, y)
        function = self._create_objective(X, y)
        return self._minimize(function)
    
    def _create_objective(self, X: np.ndarray, y: np.ndarray) -> ObjectiveFunction:
        """
        Creates an objective function that depends on the classification we want to solve.

        Args:
            X: The input data.
            y: True values for ``X``.

        Returns:
            An instance of the objective function.
        """
        # mypy definition
        function: ObjectiveFunction = None
        self._validate_binary_targets(y)
        function = MixedState_BinaryObjectiveFunction(X, y, self._neural_network, self._loss)
        return function
    