# Databricks notebook source
# MAGIC %md ## SystemDump 
# MAGIC
# MAGIC * Dump system info about Databricks.

# COMMAND ----------

# MAGIC %md ### Execute command on all workers 
# MAGIC
# MAGIC * from: Meng's [Util](https://demo.cloud.databricks.com/#notebook/1671294) notebook

# COMMAND ----------

def get_ip():
  from subprocess import check_output
  return check_output(['hostname', '-I']).decode().split(' ')[0]

def get_worker_ips():
  from subprocess import check_output
  n = sc._jsc.sc().getExecutorMemoryStatus().keys().size()
  #print("Number of workers:", n)
  ips = sc.parallelize(range(0, 2*n), 2*n)\
    .map(lambda x: get_ip())\
    .distinct()\
    .collect()
  return ips

# COMMAND ----------

get_worker_ips()

# COMMAND ----------

# MAGIC %md #### Node versions

# COMMAND ----------

import sys
workers = sc.parallelize([1], 1).mapPartitions(lambda x: [sys.version]).collect()
print("Driver:", sys.version.replace("\n"," "))
print("Workers:") 
for w in workers:
    print("   ", w)

# COMMAND ----------

# MAGIC %md ### Linux system

# COMMAND ----------

# MAGIC %sh uname -a  

# COMMAND ----------

# MAGIC %sh cat /proc/version

# COMMAND ----------

# MAGIC %sh lsb_release -a

# COMMAND ----------

# MAGIC %sh cat /etc/lsb-release

# COMMAND ----------

# MAGIC %sh cat /etc/os-release

# COMMAND ----------


