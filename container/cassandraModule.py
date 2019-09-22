import os
import sys   
import logging   
from cassandra.cluster import Cluster

KEYSPACE = "digitrecognition"

class cassandraModule:
    
    ## cluster_IP is a list of IP addresses for cassandra cluster.
    ## Todo: figuring out why it has to be list of IP if it only connects to one of 
    ## the nodes in the cluster
    ## Todo: read the list option in Cluster documentation
    ## Not sure if the session can be set ahead of operations
    def __init__(self, cluster_IP, cluster_Port = 9042):
        self.cluster = Cluster(contact_points = cluster_IP, port = cluster_Port)
        self.session = self.cluster.connect()
        self.log = logging.getLogger()
        self.log.setLevel("INFO")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        self.log.addHandler(handler)
        self.keySpace = {}
        return
    
    ## This function are implemented only for this DigitRecognition Project.
    def createKeySpace(self, key_space = KEYSPACE):
        self.log.info("Creating keyspace : %s" % key_space)
        try:
            self.session.execute(""" CREATE KEYSPACE IF NOT EXISTS %s WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '2'}""" % key_space)
            self.log.info("Setting keyspace...")
            self.session.set_keyspace(key_space)
            self.keySpace.update({key_space: []})
            self.log.info("Creating table...")
            self.creatDRTable()
            table_list = self.keySpace[key_space]
            table_list.append("dr_request_log")
            self.keySpace.update({KEYSPACE: table_list})
        except Exception as e:
            self.log.error("Unable to create keyspace.")
            self.log.error(e)
        return
    ## create table for DigitRecognition requset records.
    def creatDRTable(self):
        try:
            ## The request_time is set as text. It should be set as particular data type for time data in the future version.
            self.session.execute("""CREATE TABLE IF NOT EXISTS dr_request_log(file_name text, request_time text, result int, PRIMARY KEY(request_time))""")
        except Exception as e:
            self.log.error("Unable to create table.")
            self.log.error(e)
        return
        

    def insertRecord(self, file_name, time, result):
        
        try:
            self.session.execute("""INSERT INTO dr_request_log (file_name, request_time, result) 
                                 VALUES (%(file_name)s, %(request_time)s, %(result)s)""", {"file_name": file_name, "request_time": time, "result": result})
        except Exception as e:
            self.log.error("Unable to save request log.")
            self.log.error(e)
    
        return