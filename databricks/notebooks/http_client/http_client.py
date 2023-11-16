# Databricks notebook source
import json, os
import requests

_TIMEOUT = 15

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()

class HttpClient:
    def __init__(self, api_prefix, host=None, token=None):
        host = host or self._get_host()
        self.api_uri = os.path.join(host, api_prefix)
        self.token = token or self._get_token()

    def get(self, resource, params=None):
        uri = self._mk_uri(resource)
        rsp = requests.get(uri, headers=self._mk_headers(), json=params, timeout=_TIMEOUT)
        self._check_response(rsp)
        return rsp.json()

    def post(self, resource, data=None):
        uri = self._mk_uri(resource)
        data = json.dumps(data) if data else None
        rsp = requests.post(uri, headers=self._mk_headers(), data=data, timeout=_TIMEOUT)
        self._check_response(rsp)
        return rsp.json()

    def _mk_headers(self):
        if self.token:
            return { "Authorization": f"Bearer {self.token}" }
        return {}

    def _mk_uri(self, resource):
        return f"{self.api_uri}/{resource}"

    def _check_response(self, rsp):
        if not rsp.ok:
            raise rsp.raise_for_status()

    def _get_token(self):
        return ctx.apiToken().get()

    def _get_host(self):
        host = ctx.tags().get("browserHostName").get()
        return f"https://{host}"

    def __repr__(self):
        return self.api_uri

# COMMAND ----------

class DatabricksHttpClient(HttpClient):
    def __init__(self):
        super().__init__("api/2.0")

class Databricks21HttpClient(HttpClient):
    def __init__(self):
        super().__init__("api/2.1")

class MlflowHttpClient(HttpClient):
    def __init__(self):
        super().__init__("api/2.0/mlflow")
