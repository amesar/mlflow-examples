import pytest
import train
import register_model
import deploy_server
import common

model_uri = None
launch_container = False

@pytest.mark.run(order=1)
def test_train():
    train.run(common.experiment_name, common.data_path)

@pytest.mark.run(order=2)
def test_register_model():
    global model_uri
    model_uri = register_model.run(common.experiment_name, common.data_path, common.model_name)

@pytest.mark.run(order=3)
def test_deploy_server():
    deploy_server.run(model_uri, common.port, common.data_path, common.docker_image, launch_container)
