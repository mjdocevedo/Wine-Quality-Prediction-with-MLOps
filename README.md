# Setting up an MLOps project step by step 🚀

Welcome to the setup guide! Here, we'll outline the steps needed to configure and implement the various first stages of the MLOps pipeline. Follow along and fill in the details as you proceed through each step in the workflow_steps.ipynb notebook.

You can start by getting familiar with the architecture of the project: 

```bash
+---.dvc
|  
+---.github
|   \---workflows
+---data
|           
+---logs
|       
+---metrics
+---notebooks
|       workflow_steps.ipynb
|       
+---src
|   |   common_utils.py
|   |   config.py
|   |   config.yaml
|   |   config_manager.py
|   |   entity.py
|   |   __init__.py
|   |   
|   +---app
|   |       app.py
|   |       
|   +---data_module_def
|   |   |   data_ingestion.py
|   |   |   data_transformation.py
|   |   |   data_validation.py
|   |   |   schema.yaml
|   |   |   __init__.p
|   |           
|   +---models_module_def
|   |       model_evaluation.py
|   |       model_trainer.py
|   |       params.yaml
|   |       __init__.py
|   |       
|   +---pipeline_steps
|   |       prediction.py
|   |       stage01_data_ingestion.py
|   |       stage02_data_validation.py
|   |       stage03_data_transformation.py
|   |       stage04_model_trainer.py
|   |       stage05_model_evaluation.py
|   |       __init__.py
|           
+---templates
|       index.html
|       login.html
|       register.html
|       results.html
|       
+---users
|   |   users.json
| 
|   .dvcignore
|   .gitignore
|   custom_logger.py
|   dvc.lock
|   dvc.yaml
|   README.md
|   requirements.txt
|   tree
|   __init__.py
|  
```
Through this project we'll work with a wine dataset 🍷 The goal will be to implement a model that will predict its quality, all while adhering to the best practices in MLOps in terms of version control, use of pipelines and the most commonly used tools.

First of all you need to start by forking and cloning the project. Then, you must create a virtual environment where you'll install all the necessary libraries. These can be found in the `requirements.txt` file 📚 Make sure you activate your virtual environment before you use it 😉

For what follows, the first two steps have been provided so all you have to do is have a look at the files to make sure you understand the workflow.

## Step 1: Configuration Files 📘
Let's have a quick look at the different `yaml` files in our `src` folder.

You can start by having a look at the `config.yaml` 📂 You will see that it sets the paths to the different files that will be used in each of the steps we'll put in place.

Next, inside the `data_module_def` folder we have the `schema.yaml` 🗃️ If you have a look at it you'll see it defines the data types for each column in the dataset we'll work with.

Finally, you can have a look at `params.yaml` 📊 inside the `models_module_def` folder. What this file does is set the hyperparameters of the model we'll put in place.

⚠️ The file `src/config.py` defines the global variables containing the paths to these yaml files to facilitate their access. 

## Step 2: Common Utilities 🛠️ 
In `src/common_utils.py`  we have reusable functions:

* read_yaml(filepath: str) -> dict
* create_directories(paths: List[str])
* save_json(path: str, data: dict)
* load_json

These utilities will streamline the loading of configurations and ensure necessary directories are created.

For the next steps you can use the notebook `workflow_steps.ipynb` to guide you through the code you'll need to write on each of the corresponding files 🧑‍💻

## Step 3: Define Configuration Classes 🧩
In `src/entity.py` define `dataclasses` for configuration objects:

* DataIngestionConfig
* DataValidationConfig
* DataTransformationConfig
* ModelTrainerConfig
* ModelEvaluationConfig

These configurations will help in managing the settings and parameters required for each stage in a clean and organized manner. Refer to the corresponding cell in `workflow_steps.ipynb` for class definitions.

## Step 4: Configuration Manager 🗄️
In `src/config_manager.py`, create a class to manage configurations. This class will:

* Read paths from `config.yaml`
* Read hyperparameters from `params.yaml`
* Read the data types from `schema.yaml`.
* Create configuration objects for each of the stages through the help of the objects defined on the step before: DataIngestionConfig, DataValidationConfig, ModelTrainerConfig and ModelEvaluationConfig.
* Create necessary folders

⚠️ Pay attention to the `mlflow_uri` on the `get_model_evaluation_config`, make sure you adapt it with your own dagshub credentials. 

Again, you can refer to the notebook for full implementation details 😉

## Step 5: Data module definition and model module definition.
In the corresponding files of `src/data_module_def`, create:

1. Data Ingestion module 📥
This class will:
* Download the dataset into the appropriate folder.
* Unzip the dataset into the appropriate folder.

2. Data Validation module ✅
This class will:
* Validate columns against the schema. Optional: you can also verify the informatic type.
* Issue a text file saying if the data is valid.

3. Data Transformation module 🔄
This class will:
* Split the data into training and test sets.
* Save the corresponding csv files into the appropriate folder.

In the corresponding files of `src/models_module_def`, create:

1. Model trainer module 🏋️‍♂️
This class will:
* Train the model using the hyperparameters specified in `params.yaml`.
* Save the trained model into the appropriate folder.

2. Model Evaluation module 📝
This class will
* Evaluate the model and log metrics using MLFlow

## Step 6: Pipeline Steps 🚀
In `src/pipeline_steps`, create scripts for each stage to instantiate and run the processes:

* stage01_data_ingestion.py
* stage02_data_validation.py
* stage03_data_transformation.py
* stage04_model_trainer.py
* stage05_model_evaluation.py

## Step 7: Use DVC to connect the different stages of your pipeline
Start by setting DagsHub as your distant storage through DVC.

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/your_username/your_repo.s3 
dvc remote default origin
```

Use dvc to connect the different steps of your pipeline as follows 
```bash
dvc stage add -n data_ingestion -d wine_quality/src/pipeline_steps/stage01_data_ingestion.py -d wine_quality/src/config.yaml -o wine_quality/data/raw/winequality-red.csv python wine_quality/src/pipeline_steps/stage01_data_ingestion.py
```
You can run the pipeline through the command `dvc repro`.

Congratulations! 🎉 Now that you have a structured and well-defined MLOps project you're ready for the next step which is the creation of the API.

Each step is modularized, making it easy to maintain, extend, and scale your Machine Learning pipeline. 




