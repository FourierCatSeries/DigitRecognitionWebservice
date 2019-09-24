FROM python:3.7-slim

WORKDIR /digitRecognition

COPY . /digitRecognition

RUN pip install -U tensorflow
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME Service

RUN chmod +x webservice.py
CMD ["python", "webservice.py"]

