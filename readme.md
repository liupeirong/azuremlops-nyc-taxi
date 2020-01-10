## Predict trip duration based on NYC taxi data


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

### Data drift detection


### TODO
* Use a single docker image for running unit tests, training, and inferencing. Currently, unit tests use a conda environment, deployment installs python Azure ML packages, training uses another docker image, and inference yet another.  Even though not all modules need to be in all cases, it's much easier to maintain just one image.
* Modify the data prep pipeline to deploy to an approval-required environment.  Continuous Integration is currently disabled on it so that it doesn't generate new data every time we change code.
* Add model explanation.
* Add an Azure automl model. 