import os
from azureml.core import Workspace, Datastore, Dataset, Environment
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.model import Model
from azureml.core.webservice import AksWebservice, Webservice
from azureml.datadrift.datadriftdetector import DataDriftDetector
from azureml.datadrift import AlertConfiguration
from datetime import datetime, timedelta


def setup_azureml():
    """
    Get an Azure ML workspace from environment variables.
    Assumes the following are created outside of the code in this project:
      AML workspace
      AML datastore
      AML compute resource for training (can be blank for inferencing)
      AML compute resource for inferencing (can be blank for training)
    """
    subscription_id = os.environ['AML_SUBSCRIPTION']
    resource_group = os.environ['AML_RESOURCE_GROUP']
    workspace_name = os.environ['AML_WORKSPACE']
    datastore_name = os.environ['AML_DATASTORE']
    training_target_name = os.environ.get('AML_COMPUTE')
    inference_target_name = os.environ.get('AML_INFERENCE_COMPUTE')
    ws = Workspace(subscription_id, resource_group, workspace_name)
    ds = Datastore.get(ws, datastore_name=datastore_name)
    if training_target_name is not None:
        training_target = ws.compute_targets[training_target_name]
    else:
        training_target = None
    if inference_target_name is not None:
        inference_target = ws.compute_targets[inference_target_name]
    else:
        inference_target = None
    return ws, ds, training_target, inference_target


def create_azureml_env(ws, env_name, conda_yml):
    """
    Create an Azure ML environment based a default AML docker image
    and a yaml file that specifies Conda and Pip dependencies.
    Azure ML will create a new custom docker image for the env.
    """
    try:
        amlenv = Environment.get(ws, name=env_name)
        print('found existing env {}'.format(amlenv.name))
    except Exception:
        print('create new env {}'.format(env_name))
        amlenv = Environment.from_conda_specification(
            name=env_name, file_path=conda_yml)
        amlenv.docker.enabled = True
        amlenv.docker.base_image = DEFAULT_CPU_IMAGE
        amlenv.python.user_managed_dependencies = False
        amlenv.register(ws)
    return amlenv


def register_model(run, datastore, data_folder, model_name):
    """
    register a model from a run and the training dataset with the model
    so that we can do data drift detection later. Assumes model
    file is uploaded to the outputs folder in AML
    """
    tabular_train_dataset = Dataset.Tabular.from_delimited_files(
        path=[(datastore, data_folder)])

    # model already keeps run info, no need to tag it
    model = run.register_model(
        model_path=os.path.join('outputs', model_name),
        model_name=model_name,
        datasets=[(Dataset.Scenario.TRAINING, tabular_train_dataset)])

    return model


def deploy_service(ws, model, inference_config,
                   service_name, compute_target):
    tags = {'model': '{}:{}'.format(model.name, model.version)}

    try:
        service = Webservice(ws, service_name)
        print("Service {} exists, update it".format(service_name))
        service.update(
            models=[model],
            inference_config=inference_config,
            tags=tags)
    except Exception:
        print('deploy a new service {}'.format(service_name))
        deployment_config = AksWebservice.deploy_configuration(
            cpu_cores=1, memory_gb=2, tags=tags,
            collect_model_data=True, enable_app_insights=True)
        service = Model.deploy(
            ws, service_name, [model],
            inference_config, deployment_config, compute_target)

    service.wait_for_deployment(show_output=True)

    if service.auth_enabled:
        token = service.get_keys()[0]
    elif service.token_auth_enabled:
        token = service.get_token()[0]

    return service.scoring_uri, token


def create_data_drift_detector_for_model(
        ws, model, service_name, compute_name,
        feature_list, alert_email_list, drift_threshold):
    services = [service_name]
    alert_config = AlertConfiguration(alert_email_list)
    start = datetime.utcnow() - timedelta(days=1)

    try:
        monitor = DataDriftDetector.get(ws, model.name, model.version)
        print('data drift detector for {}.{} already exists'.format(
            model.name, model.version))
    except Exception:
        print('create data drift detector for {}.{}'.format(
            model.name, model.version))
        monitor = DataDriftDetector.create_from_model(
            ws, model.name, model.version, services,
            frequency='Day',
            schedule_start=start,
            alert_config=alert_config,
            drift_threshold=drift_threshold,
            compute_target=compute_name)

    return monitor
