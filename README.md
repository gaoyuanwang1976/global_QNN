run global QNN with 50 epochs, patience 10, and 10 batches with random batching: python QClassifier_AllStates.py -i [your_dataset] -e 50 --global_state --num_batches 10 --output /output -p 10
run global QNN with 50 epochs, patience 10, and 10 batches with smart batching: python QClassifier_AllStates.py -i [your_dataset] -e 50 --smart_batch --global_state --num_batches 10 --output /output -p 10
run instance-level QNN with 50 epochs, patience 10: python QClassifier_AllStates.py -i [your_dataset] -e 50 --output /output -p 10
