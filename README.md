# mitRemoteProject
---
This project developed a hand-written digit recognition service with deep convolutional neural network digit recognizer and Flask web service and a Cassandra database to store the service request records. The service is deployed in docker container for portability and convenience.
---
## Componets
---
### hand-written digit recognizer
---
#### Requirements:
---
- [x] Python 3
- [x] TensorFlow environment [Install TensorFlow](https://www.tensorflow.org/install)
- [x] OpenCV 2 [Install OpenCV](https://github.com/opencv/opencv)
---
#### Prediction.py
---
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
pre-process the image to be recognized by:
```
processed, original= Prediction.load_input_image(path_to_input_image)
```
The output```original``` is the ```cv2``` image object of the original image. ```processed``` is the processed image file through the procesdure of ```converting to grayscale -> resizing to 28 pixels by 28 pixels -> reshaped to (1,784) tensor```. 

Then recognize the image by:
```
recognizer.Predict_Label(processed)
```
---
### Cassandra database module
---
#### Requirements
---
- [x] Python 3
- [x] cassandra-driver
install cassandra-driver by;
```
pip install cassandra-driver
```
or 
```
pip3 install cassandra-driver
```
---
#### cassandraModule.py
---
initiate cassanraModule object by
```
cas = cassandraModule.cassandraModule(cluster_IP = [list of IP address of cassandra cluster IP], cluster_port = port_to_communicate_cassandra_cluster)
```
You want to have a cassandra cluster launched before initialize cassandraModule object. Please refer to [Launch Cassandra Cluster in container](####-launch-Cassandra-Cluster-in-container)

create keyspace in the cluser by:
```
cas.createKeySpace(name_of_keyspace)
```
The function ```creatDRTable``` and ```insertRecord``` are coded specific for digit recognition service request records.

insert the record to the talbe by:
```
cas.inserRecord(file_name, time_of_request, recognized_digit)
```
---
### Flask service
---
#### Requirements
---
- [x] Flask
install by:
```
pip install Flask
```
---
#### webservice.py ####
---
The flask webservice has RESTful API of method POST listening to localhost:5000/digitRecognition/localService/v1.0.
```webservice.py``` module in the root directory of the repository call ```Prediction.py``` and ```cassandraModule.py```setting IP address to ```CASSANDRA_CLUSTER_IP``` and communication port to ```CASSANDRA_CLUSTER_PORT```of cassandra cluster in the header of ```webservice.py```
Run the service in your host machine by
```
python webservice.py
```
or
```
python3 webservice.py
```
---
use the service by using ```curl``` command like following:
```
curl -X POST -F "file=@path_to_image_file" localhost:5000/digitRecognition/localService/v1.0
```
## Service in Container

the service is also deployed into docker image.
navigate to ```/container``` directory and build the image by:
```
docker build -t name_of_image .
```
The period ```.``` at the end of the command is necessary.
Create a docker network which will be used by both the flask service container and the cassandra cluster container:
```
docker network create name-network
```
launch the container with interactive shell so that you can monitor the behavior of the flask server.
```
docker run --name=name_of_container -v 

### Cassandra cluster container
---
#### Requirements
---
- [x] Docker [Install Docker](https://docs.docker.com/install/)
---
#### Pull Cassandra image
---
use docker command to get Cassandra image:
```
docker pull cassandra
```
---
#### Launch Cassandra Cluster in container
---

