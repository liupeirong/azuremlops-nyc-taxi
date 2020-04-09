CI with Lint and unit test: [![Lint & test](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/nyctaxi_ci?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=10&branchName=master)

Data preparation: [![Data prep](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/nyctaxi_data_prep?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=11&branchName=master)

Train and deploy model: [![ML train & deploy](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/nyctaxi_ml?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=12&branchName=master)

Build AutoML training pipeline: [![Build pipeline](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/automl_create_pipelineendpoint?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=13&branchName=master)

Train with published AutoML pipeline: [![Run pipeline](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/train_on_automl_pipelineendpoint?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=14&branchName=master)


## Predict trip duration based on NYC taxi data

This example demonstrates the following use cases:
1. Continuous integration with Azure DevOps pipeline to do [linting and unit testing](devops_pipelines/azure-pipelines-ci.yml) on machine learning code. 
2. MLOps with Azure DevOps pipelines to 
    * [prepare data](devops_pipelines/azure-pipelines-data_prep.yml) to upload to Azure ML data store 
    * [submit model to train in Azure ML compute instance](devops_pipelines/azure-pipelines-ml.yml), register and deploy the model to Azure Kubernetes Service upon [approval](https://docs.microsoft.com/en-us/azure/devops/pipelines/process/approvals?view=azure-devops&tabs=check-pass), and verify if the deployed service is healthy
3. [Build an Azure ML pipeline](devops_pipelines/azure-pipelines-automl-build-pipeline.yml) to train the model using AutoML. [Trigger the pipeline](devops_pipelines/azure-pipelines-automl-train.yml) to run AutoML training. 
4. [Interpret AutoML model](notebooks/nyc_automl.ipynb) at the global level as well as locally for each inference data point. 
5. [Detect data drift](notebooks/nyc_lgbm.ipynb) on deployed model. 

### Project structure
* [code](code): python code for data processing, training, and scoring
* [ml_service](ml_service): python code for communicating with AzureML for model training and deployment
* [devops_pipelines](devops_pipelines): Azure DevOps pipelines to run code linting and unit testing, as well as using the code in ml_service to do model training, deployment, and verification
* [notebooks](notebooks): exploration notebooks for model interpretability and data drift detection
