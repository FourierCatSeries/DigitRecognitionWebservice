# mitRemoteProject

This project developed a hand-written digit recognition service with deep convolutional neural network digit recognizer and Flask web service and a Cassandra database to store the service request records. The service is deployed in docker container for portability and convenience.
---
## hand-written digit recognizer
The recognizer was trained with mnist dataset and achieved test accuracy of 93.2%. The trained model was saved to checkpoint files in the directory of /data/trained_models. The checkpoint files were saved in the following way:
```
model_saver = TensorFlow.train.Saver()
model_saver.save(some_tensorflow_session, path_to_saved_model)
```
The recognizer model was wrapped up as DigitRecognizer class in ```Prediction.py``` and ```Prediction_r.py``` modules. ```Prediction.py``` is for local using while ```Prediction_r.py``` is for envrionment inside the container. You can initialize a instance of DigitRecognizer object which has the same structure of the deep CNN module used in the project by statment below:
```
recognizer = Prediction.DigitRecognizer()
```
 Then you can load any pre-trained model saved in the same fashion as the projct from checkpoint files by:
```
recognizer.Load_Model(path_to_directory_model_saved, model_name)
```
