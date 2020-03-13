#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import nested_scopes
#import findspark
#findspark.init()


# In[2]:


import re
import os
import pandas
pandas.set_option('display.max_rows', None)


# In[3]:


import threading
import collections
import gzip


# In[4]:


import pyspark
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, DateType,
    TimestampType, StringType, LongType, IntegerType, DoubleType,FloatType)
from pyspark.sql.functions import to_date, floor
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit
import time, timeit
from pyspark.storagelevel import StorageLevel
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col


# In[5]:


from pyspark.ml import Pipeline
import pandas
import numpy as np


# In[6]:


import math
from functools import reduce
import json


# In[7]:


from pyspark.sql.types import *
from pyspark.sql import functions as F
from datetime import date


# In[8]:

#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/yuzhou/.m2/repository/ml/dmlc/xgboost4j-spark_2.11/distr_opt/xgboost4j-spark_2.11-distr_opt.jar,/home/yuzhou/.m2/repository/ml/dmlc/xgboost4j_2.11/distr_opt/xgboost4j_2.11-distr_opt.jar pyspark-shell'
#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/yuzhou/.m2/repository/ml/dmlc/xgboost4j-spark_2.11/1.0.0-SNAPSHOT/xgboost4j-spark_2.11-1.0.0-SNAPSHOT.jar,/home/yuzhou/.m2/repository/ml/dmlc/xgboost4j_2.11/1.0.0-SNAPSHOT/xgboost4j_2.11-1.0.0-SNAPSHOT.jar pyspark-shell'
#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/yuzhou/.m2/repository/ml/dmlc/xgboost4j-spark/0.82/xgboost4j-spark-0.82.jar,/home/yuzhou/.m2/repository/ml/dmlc/xgboost4j/0.82/xgboost4j-0.82.jar pyspark-shell'
#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/yuzhou/.m2/repository/ml/dmlc/xgboost4j-spark_2.11/1.0.0-SNAPSHOT-master-pr-4824/xgboost4j-spark_2.11-1.0.0-SNAPSHOT-master-pr-4824.jar,/home/yuzhou/.m2/repository/ml/dmlc/xgboost4j_2.11/1.0.0-SNAPSHOT-master-pr-4824/xgboost4j_2.11-1.0.0-SNAPSHOT-master-pr-4824.jar pyspark-shell'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master yarn --jars /home/xgboost/.m2/repository/ml/dmlc/xgboost4j-spark_2.12/1.1.0-SNAPSHOT/xgboost4j-spark_2.12-1.1.0-SNAPSHOT.jar,/home/xgboost/.m2/repository/ml/dmlc/xgboost4j_2.12/1.1.0-SNAPSHOT/xgboost4j_2.12-1.1.0-SNAPSHOT.jar --conf "spark.executor.extraLibraryPath=/home/xgboost/install/OneCCL/oneccl/build/_install/lib" pyspark-shell'

#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /home/xgboost/.m2/repository/ml/dmlc/xgboost4j/0.82/xgboost4j-0.82.jar,/home/xgboost/.m2/repository/ml/dmlc/xgboost4j/1.0.0-SNAPSHOT/xgboost4j_2.12-1.0.0-SNAPSHOT.jar pyspark-shell'

#/home/xgboost/.m2/repository/ml/dmlc/xgboost4j/0.82/xgboost4j-0.82.jar
#Dev
#os.environ['OMP_NUM_THREADS'] = '72'

# In[22]:


RANDOM_SEED = 42
clients = ["sr243"]

# In[9]:

# To run on multiple nodes:
'''executors_per_node = 7
nodes=len(clients)
cores_per_executor=8
task_per_core=8
'''
# In[10]:

# to test:
nodes=1
executors_per_node = 2
cores_per_executor = 1
task_per_core = 1

cache_size=50
total_size=340000
print('executor per node: {:d}\nparallelism: {:d}\nmemory: {:d}m\noffheap:{:d}m'.format(executors_per_node,nodes*executors_per_node*cores_per_executor*task_per_core,int(math.floor(nodes*total_size/(nodes*executors_per_node)))-1024-int(math.floor(cache_size*1024/(nodes*executors_per_node))),int(math.floor(cache_size*1024/(nodes*executors_per_node)))))


# In[11]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


# In[40]:

conf = SparkConf()\
    .set('spark.default.parallelism', '{:d}'.format(nodes*executors_per_node*cores_per_executor*task_per_core))\
    .set('spark.executor.instances', '{:d}'.format(executors_per_node*nodes))\
    .set('spark.files.maxPartitionBytes', '256m')\
    .set('spark.app.name', 'pyspark_final-xgboost-0.90-DMLC')\
    .set('spark.rdd.compress', 'False')\
    .set('spark.serializer','org.apache.spark.serializer.KryoSerializer')\
    .set('spark.executor.cores','{:d}'.format(cores_per_executor))\
    .set('spark.executor.memory', '{:d}m'.format(int(math.floor(nodes*total_size/(nodes*executors_per_node)))-1024-int(math.floor(cache_size*1024/(nodes*executors_per_node)))))\
    .set('spark.task.cpus','{:d}'.format(cores_per_executor))\
    .set('spark.driver.memory','24g')\
    .set('spark.memory.offHeap.enabled','True')\
    .set('spark.memory.offHeap.size','{:d}m'.format(int(math.floor(cache_size*1024/(nodes*executors_per_node)))))\
    .set('spark.executor.memoryOverhead','{:d}m'.format(int(math.floor(cache_size*1024/(nodes*executors_per_node)))+3000))\
    .set('spark.sql.join.preferSortMergeJoin','False')\
    .set('spark.memory.storageFraction','0.5')\
    .set('spark.executor.extraJavaOptions','-XX:+UseParallelGC -XX:+UseParallelOldGC -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -DCCL_ATL_TRANSPORT=ofi -DCCL_WORLD_SIZE=1 -DCCL_PM_TYPE=resizable -DCCL_KVS_IP_EXCHANGE=env -DCCL_KVS_IP_PORT=10.0.0.143_9877 -DWORK_DIR=/home/xgboost/install/OneCCL/oneccl/build/_install/env -DCCL_ROOT=/home/xgboost/install/OneCCL/oneccl/build/_install -DI_MPI_ROOT=/home/xgboost/install/OneCCL/oneccl/build/_install -DCCL_ATL_TRANSPORT_PATH=/home/xgboost/install/OneCCL/oneccl/build/_install/lib -DFI_PROVIDER_PATH=/home/xgboost/install/OneCCL/oneccl/build/_install/lib/prov')\
    .set('spark.driver.maxResultSize', 0)\
    .set('spark.eventLog.dir', '/home/yuzhou/spark_local')\
    .set('spark.executor.extraLibraryPath', '/home/xgboost/install/OneCCL/oneccl/build/_install/lib')\
    .set('spark.driver.extraClassPath', '/home/xgboost/.m2/repository/ml/dmlc/xgboost4j_2.12/1.1.0-SNAPSHOT/xgboost4j_2.12-1.1.0-SNAPSHOT.jar')\
    .set('spark.executor.extraClassPath', '/home/xgboost/.m2/repository/ml/dmlc/xgboost4j_2.12/1.1.0-SNAPSHOT/xgboost4j_2.12-1.1.0-SNAPSHOT.jar')\
    .setExecutorEnv('CCL_PM_TYPE','resizable')\
    .setExecutorEnv('CCL_ATL_TRANSPORT','ofi')\
    .setExecutorEnv('CCL_KVS_IP_EXCHANGE','env')\
    .setExecutorEnv('CCL_KVS_IP_PORT','10.0.0.143_9877')\
    .setExecutorEnv('CCL_ROOT','/home/xgboost/install/OneCCL/oneccl/build/_install')\
    .setExecutorEnv('CCL_WORLD_SIZE',1)\
    .setExecutorEnv('I_MPI_ROOT','/home/xgboost/install/OneCCL/oneccl/build/_install')\
    .setExecutorEnv('CCL_ATL_TRANSPORT_PATH','/home/xgboost/install/OneCCL/oneccl/build/_install/lib')\
    .setExecutorEnv('FI_PROVIDER_PATH','/home/xgboost/install/OneCCL/oneccl/build/_install/lib/prov')
'''
conf = SparkConf()    .set('spark.default.parallelism',
        f'{nodes*executors_per_node*cores_per_executor*task_per_core}')
    .set('spark.executor.instances', '{:d}'.format(executors_per_node*nodes))
.set('spark.files.maxPartitionBytes', '256m')    .set('spark.app.name',
'pyspark_final-xgboost-0.90-DMLC')    .set('spark.rdd.compress', 'False')
.set('spark.serializer','org.apache.spark.serializer.KryoSerializer')
.set('spark.executor.cores','{:d}'.format(cores_per_executor))
.set('spark.executor.memory',
'{:d}m'.format(int(math.floor(nodes*total_size/(nodes*executors_per_node)))-1024-int(math.floor(cache_size*1024/(nodes*executors_per_node)))))
.set('spark.task.cpus',f'{cores_per_executor}')
.set('spark.driver.memory','24g')    .set('spark.memory.offHeap.enabled','True')
.set('spark.memory.offHeap.size','{:d}m'.format(int(math.floor(cache_size*1024/(nodes*executors_per_node)))))
.set('spark.executor.memoryOverhead','{:d}m'.format(int(math.floor(cache_size*1024/(nodes*executors_per_node)))+3000))
.set('spark.sql.join.preferSortMergeJoin','False')
.set('spark.memory.storageFraction','0.5')
.set('spark.executor.extraJavaOptions',         '-XX:+UseParallelGC
        -XX:+UseParallelOldGC -verbose:gc -XX:+PrintGCDetails
        -XX:+PrintGCTimeStamps')    .set('spark.driver.maxResultSize', 0)
.set('spark.eventLog.dir', '/home/yuzhou/spark_local')
'''
#.set('spark.kryoserializer.buffer.max', '2048m')
#spark.driver.maxResultSize it was "3g". Now it is unlimited 


# In[42]:


#sc.stop()
#sc = SparkContext(conf=conf,master='yarn')
# To run on local node, single node distributed mode:
sc = SparkContext(conf=conf,master='yarn')
sc.setLogLevel('INFO')
spark = SQLContext(sc)
sc.addPyFile('/home/xgboost/install/xgb/sparkxgb_0.83.zip')
time.sleep(10)


# In[ ]:


# loading and splitting data:

# replication != 1 so it is NOT guaranteed that each partition receives the same data
#df = spark.read.format('parquet').load('hdfs://sr507/user/yuzhou/xgboost_36_files.parquet')
#df = spark.read.format('parquet').load('hdfs://sr507/user/yuzhou/xgboost_3.5G.parquet') # was using this one
# replication=1 so it is guaranteed that each partition receives the same data
#df = spark.read.format('parquet').load('hdfs://sr507/user/yuzhou/xgboost_36_files_1rep.parquet')

# For comparison agains Rabit single-node without Spark:
df = spark.read.format('parquet').load('hdfs://10.1.0.143:9000//smallin/inputData/')

#For PR4824 (09/09/19):
#df = spark.read.format('parquet').load('hdfs://sr507/user/yuzhou/xgboost_36_files.parquet')
#print("Input DF numPartitions = {:d}".format(df.rdd.getNumPartitions()))
df = df.coalesce(executors_per_node*nodes*cores_per_executor)
#print(df.count())

#224 partitions

print('Completed data loading.')
'''
(trainingData, testData) = df.randomSplit([0.9, 0.1], seed = RANDOM_SEED)
trainingData=trainingData.coalesce(executors_per_node*nodes*cores_per_executor)
print('Completed coalesce in training data.')
testData=testData.coalesce(executors_per_node*nodes*cores_per_executor)
print('Completed coalesce in test data.')

trainingData.cache()
testData.cache()
print('trainingData count:', trainingData.count())
print('testData count:', testData.count())

######
print('Completed data spliting.')

# In[ ]:


(tr2, te2) = testData.randomSplit([0.9999, 0.0001], seed = RANDOM_SEED)
tr2=tr2.coalesce(executors_per_node*nodes*cores_per_executor)
print('te2 data count:', te2.count())
'''

# In[ ]:


from sparkxgb import XGBoostClassifier


# # Save and load model

# In[33]:


def run_train_orig(train_data):
    t1 = time()
    xgboost = XGBoostClassifier(
        featuresCol="features",
        labelCol="delinquency_12", 
        numRound=100,
        maxDepth=8,
        maxLeaves=256,
        alpha=0.9,
        eta=0.1,
        gamma=0.1,
        subsample=1.0,
        reg_lambda=1.0,
        scalePosWeight=2.0,
        minChildWeight=30.0,
        treeMethod='hist',
        objective='reg:linear', #squarederror', #if xgboost v0.82 needs to use 'reg:linear'
        growPolicy='lossguide', #depthwise
        numWorkers=executors_per_node*nodes*cores_per_executor,
        nthread=1,
        #evalMetric='logloss' # mconrado added that to test. The log loss is only defined for two or more labels.
    )

    model = xgboost.fit(train_data)
    t2 = time()
    print(f"Training total time: {t2-t1}")
    return model

def run_train(train_data, params):
    t1 = timeit.default_timer()
    '''    xgboost = XGBoostClassifier(
        featuresCol='features',
        labelCol='delinquency_12',
        numRound=20,
        maxDepth=8,
        maxLeaves=256,
        alpha=0.9,
        eta=0.1,
        gamma=0.1,
        subsample=1.0,
        reg_lambda=1.0,
        scalePosWeight=2.0,
        minChildWeight=30.0,
        treeMethod='hist',
        objective='reg:linear', #squarederror', #if xgboost v0.82 needs to use 'reg:linear'
        growPolicy='lossguide',
        numWorkers=executors_per_node*nodes,
        nthread=cores_per_executor
    )
    '''
    xgboost =  XGBoostClassifier(**params)
    model = xgboost.fit(train_data)
    t2 = timeit.default_timer()
    train_time = t2 - t1
    return model, train_time

import sklearn.metrics as metrics
from sklearn.metrics import auc
def run_predict(model, test_data):
    t1 = time()
    results = model.transform(test_data)
    t2 = time()
    predict_time = t2-t1

    t1 = time()
    preds = results.select(results.prediction).collect()
    Y_test = results.select(results.delinquency_12).collect()
    auc = metrics.roc_auc_score(Y_test, preds)
    t2 = time()
    #print(f"Conversion data + AUC calculation time: {t2-t1}")
    conversion_auccalculation_time = t2-t1

    return auc, predict_time, conversion_auccalculation_time

def calc_print_results(spent_time, preds, Y, msg='Results'):
    if (preds>1).any()==True:
        print('W: It seems predicted values are probabilities. Please convert them.')    
    err = 1 - metrics.accuracy_score(Y, preds)
    auc = metrics.roc_auc_score(Y, preds)
    print('{}: \t\t {} \t {} \t {}'.format(msg, err, auc, spent_time))

from sklearn.preprocessing import LabelEncoder 
from time import time
def reeval_saved_model(X, Y, model_path, msg='Results'):
    loaded_model = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model_path)
    loaded_model._Booster = booster
    loaded_model._le = LabelEncoder().fit(np.unique(Y))
    #preds_proba = loaded_model.predict_proba(X)
    t1 = time()
    preds = loaded_model.predict(X)
    calc_print_results(time()-t1, preds, Y, msg)
    print('Reevaluation of saved model completed:', model_path)
    
def save_model(model, model_path):
    model.nativeBooster.saveModel(model_path)

def save_model_txt(model, model_path):
    dump = model.nativeBooster.getModelDump('', False, 'text')
    with open(model_path, 'w+') as f:
        for s in dump:
            f.write(s + '\n')

# Convert pyspark.sql.dataframe.DataFrame TO numpy array
def convert_dfspark_2_nparray(dfspark):
    #pandas_df_sample = te2.limit(2).select('delinquency_12').toPandas()
    X_pandas_df_sample = dfspark.select('features').toPandas() # trainingData count: 546040147
    Y_pandas_df_labels = dfspark.select('delinquency_12').toPandas()
    
    X_ndarray = np.array(X_pandas_df_sample['features'].tolist())
    print('Instance:', X_ndarray[0])
    Y_ndarray = np.array(Y_pandas_df_labels['delinquency_12'].tolist())
    print('Label:', Y_ndarray[0])

    print('Size of Y_ndarray:', Y_ndarray.size)
    return X_ndarray, Y_ndarray


# In[35]:


#X_ndarray, Y_ndarray = convert_dfspark_2_nparray(tr2)


# In[ ]:

#sc.setLogLevel('ALL')

overall_start_time = timeit.default_timer()
params =    {'featuresCol': "features",
            'labelCol': "delinquency_12",
            'numRound': 100,
            'maxDepth': 8,
            'maxLeaves': 256,
          'alpha': 0.9,
          'eta': 0.1,
          'gamma': 0.1,
          'subsample': 1.0,
          'reg_lambda': 1.0,
          'scalePosWeight': 2.0,
          'minChildWeight': 30.0,
          'treeMethod': 'hist',
          'objective': 'reg:squarederror', #if xgboost v0.82 needs to use 'reg:linear'. If >= 0.9, uses squarederror
          'growPolicy': 'lossguide',  #depthwise
          'numWorkers': executors_per_node*nodes,
#          'nthread': task_per_core
          'nthread':1
#          'verbosity': 3
}
print("XGBoost Parameters: \n", params)

nRuns = 1
for i in range(0, nRuns):
    model, train_time  = run_train(df, params) #trainingData, params
    print('Completed training the model. Time(sec): ', train_time)

    # Save model as binary format
'''
    model_path = f"/home/yuzhou/notebook/mconrado/results/spark_1node_xgb_trainingData_testData_1scp1thread_sep27_{i}.modelbin"
    model_path_txt = f"/home/yuzhou/notebook/mconrado/results/spark_1node_xgb_trainingData_testData_28cpu_1ex28cor_determiFalse_oct1_{i}.txt"
#    save_model(model, model_path)
    save_model_txt(model, model_path_txt)
    print('Saving model step completed.')
    
    auc, predict_time, conversion_auccalculation_time = run_predict(model, testData)
    
    #reeval_saved_model(X_ndarray, Y_ndarray, model_path, 'Using different data to train and test')
    print(auc, ",", train_time, ",", predict_time)
    
print("AUC, training time, prediction time")
'''
print('Overall time for {} runs: {}'.format(nRuns, timeit.default_timer() - overall_start_time))


