# mitRemoteProject
---
This project developed a hand-written digit recognition service with deep convolutional neural network digit recognizer and Flask web service and a Cassandra database to store the service request records. The service is deployed in docker container for portability and convenience.
[Please refer to the video demo]()

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
docker pull cassandra:latest
```
---
#### Launch Cassandra Cluster in container
---
1. create a docker network:
```
docker network create network_name
```
2. launch the container in background
```
docker run --name=cas -p 9042:9042 --network=network_name -d cassandra:latest
```
9042 is the default port for cassandra cluster communication
3. use cqlsh to query the database
```
docker exec -it cas cqlsh
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
---
## Service in Container

the service is also deployed into docker image.
1. Navigate to ```/container``` directory and build the image by:
```
docker build -t image_name .
```
The period ```.``` at the end of the command is necessary.
2. create a docker volume to stores pre-trained models and user uploaded images.
```
docker volume create volume_name
```
you can find the mount point of this volume by:
```
docker volume inspect volume_name
```
you can also use mounted directory as volume. 
if you are using docker desktop, make sure to set the drive which contains the directory you want to use on your host machine to be shared to container in docker settings
<img src = "https://github.com/FourierCatSeries/DigitRecognitionWebservice/blob/master/shared_folder.png" />
Copy all the content under ```/data``` to the mounted point of your volume or under your shared directory.

3. [Launch Cassandra Cluster in container](####-launch-Cassandra-Cluster-in-container)

4. launch the container with interactive shell so that you can monitor the behavior of the flask server.
if you are using volume
```
docker run --name=name_of_container -v volume_name:/digitRecognition/data -p 5000:80 --network=network_name -it image_name bash
```
if you share a directory to the container
you want to set the network the same as that cassandra container uses so that the two container can communicate through the localhost network.
```
docker run --name=name_of_container -v path_to_your_directory:/digitRecognition/data -p 5000:80 --network=network_name -it image_name bash
```
5. request the digit recognition service by using ```curl``` command:
```
curl -X POST -F "file=@path_to_image_file" localhost:5000/digitRecognition/localService/v1.1
```
In this command you want to use the actural IPv4 address of your host machine as the localhost. You can find the IPv4 address by ```ifconfig``` or ```ipconfig```.
6. You can query the records of the service by ```cqlsh``` in cassandra container. The table ```dr_request_log``` carries the records and it's under the keyspace ```digitrecognition```.

---

## Video Demo

Please refer the video demo for running the service outside the container to ```localService_demo.mp4``` and video demo for running the service inside the conatiner to ```docker_service.mp4``` in the root directory of this repository.



