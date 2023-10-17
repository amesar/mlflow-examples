# Databricks notebook source
# MAGIC %run ./HttpClient

# COMMAND ----------

class ModelServingClient:

    def __init__(self):
        self.databricks_client = DatabricksHttpClient()

    def get_endpoint(self, endpoint_name):
        try:
            endpoint = self.databricks_client.get(f"serving-endpoints/{endpoint_name}")
            return endpoint
        except HttpException as e:
            if e.http_status_code != 404:
                raise e
        return None

    def start_endpoint(self, spec):
        return self.databricks_client.post("serving-endpoints", spec)

    def stop_endpoint(self, endpoint_name):
        try:
            self.databricks_client.delete(f"serving-endpoints/{endpoint_name}")
            return True
        except HttpException as e:
            if e.http_status_code != 404:
                raise e
        return False

    def list_endpoints(self):
        endpoints = self.databricks_client.get("serving-endpoints")
        if len(endpoints) > 0:
            endpoints = endpoints["endpoints"]
        return endpoints
    
    def wait_until(self, endpoint_name, max=20, sleep_time=2):
        import time
        for i in range(0,max):
            endpoint = self.get_endpoint(endpoint_name)
            if not endpoint:
                return {}
            #  'state': {'ready': 'READY', 'config_update': 'NOT_UPDATING'},
            state = endpoint.get("state",None)
            now = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime(time.time()))
            print(f"{now}: Waiting {i+1}/{max}: {state}")
            if state["ready"] == "READY" or state["config_update"] == "UPDATE_FAILED":
                return state
            import time
            time.sleep(sleep_time)
        return {}

# COMMAND ----------

model_serving_client = ModelServingClient()
