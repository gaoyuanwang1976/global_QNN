from __future__ import annotations
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives import BaseEstimator, BaseEstimatorV1, Estimator, EstimatorResult
from qiskit.quantum_info import Statevector,SparsePauliOp,DensityMatrix
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

class Test_EstimatorQNN(EstimatorQNN):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

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
        input_, shape = self._validate_input(input_data)
        weights_ = self._validate_weights(weights)
        output_data = self._forward(input_, weights_)
        output_data= (output_data+1)/2
        return output_data

class Test_NNClassifier(NeuralNetworkClassifier):
    def __init__(self,**kwargs):
        super(Test_NNClassifier, self).__init__(**kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X, _ = self._validate_input(X)

        if self._neural_network.output_shape == (1,):
            # Binary classification
            raw_output = self._neural_network.forward(X, self._fit_result.x)
            raw_output=raw_output*2-1 #Change the neural network output from (0,1) to (-1,1) for binary classification, use in combination with MixedState_EstimatorQNN and 'absolute error'
            predict = np.sign(raw_output)

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