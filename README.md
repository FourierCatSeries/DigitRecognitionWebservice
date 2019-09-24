# mitRemoteProject
This project developed a hand-written digit recognition service with deep convolutional neural network digit recognizer and Flask web service and a Cassandra database to store the service request records. The service is deployed in docker container for portability and convenience.

## hand-written digit recognizer
The recognizer was trained with mnist dataset and achieved test accuracy of 93.2%. The trained model was saved to checkpoint files in the directory of /data/trained_models. 
In prediction.py

