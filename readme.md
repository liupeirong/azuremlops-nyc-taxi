[![Lint & test](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/nyctaxi_ci?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=10&branchName=master)

[![Data prep](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/nyctaxi_data_prep?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=11&branchName=master)

[![ML train & deploy](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_apis/build/status/nyctaxi_ml?branchName=master)](https://dev.azure.com/paigedevops/azuremlops-nyc-taxi/_build/latest?definitionId=12&branchName=master)

## Predict trip duration based on NYC taxi data

### Prereq
* Create an Azure ML Workspace
* Create a training compute resource and an AKS based inference compute resource in the workspace
* Create a Azure ML DataStore with raw data located in the "input" folder
* Set up Azure DevOps variables

### Project structure
* code
* ml_service
* devops_pipelines
* notebooks

### CI/CD
* linting
* unit tests
* prepare data
* training
* deployment
Approvals: https://docs.microsoft.com/en-us/azure/devops/pipelines/process/approvals?view=azure-devops&tabs=check-pass

### Data drift detection


### TODO
* Use a single docker image for running unit tests, training, and inferencing. Currently, unit tests use a conda environment, deployment installs python Azure ML packages, training uses another docker image, and inference yet another.  Even though not all modules need to be in all cases, it's much easier to maintain just one image.
* Add model explanation.
* Add an Azure automl model. 