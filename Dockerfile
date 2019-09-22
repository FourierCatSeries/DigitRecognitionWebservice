FROM tensorflow/tensorflow

WORKDIR /digitRecognition

COPY . /digitRecognition

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME Service

CMD ["python", "webservice.py"]

