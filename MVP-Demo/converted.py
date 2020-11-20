# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Copyright (c) Microsoft Corporation. All rights reserved.
# %% [markdown]
# # Tutorial: Train your first model
# %% [markdown]
# ## Connect to Workspace

# %%
from azureml.core import Workspace
workspace = Workspace.from_config()

# %% [markdown]
# ## Create an experiment

# %%
from azureml.core import Experiment
experiment = Experiment(workspace, "diabetes-expr")

# %% [markdown]
# ## Load data and prepare for training

# %%
from azureml.opendatasets import Diabetes
from sklearn.model_selection import train_test_split

x_df = Diabetes.get_tabular_dataset().to_pandas_dataframe().dropna()
y_df = x_df.pop("Y")

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=66)

# %% [markdown]
# ## Explore Data

# %%
x_df.head()


# %%
print(x_df.dtypes)
x_df.describe()

# %% [markdown]
# ## Train a model on Notebooks

# %%
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import math

alphas = [0.1, 2, 0.3, 4, 0.5, 6, 0.7, 8, 0.9, 1.0]

for alpha in alphas:
    run = experiment.start_logging()
    run.log("alpha_value", alpha)
    print("alpha_value", alpha )
    run = experiment.start_logging()
    model = Ridge(alpha=alpha)
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)
    rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
    # run.log("rmse", rmse)
    print("rmse", rmse)
    
    model_name = "model_alpha_" + str(alpha) + ".pkl"
    filename = "outputs/" + model_name
    
    joblib.dump(value=model, filename=filename)
    
    run.upload_file(name=model_name, path_or_stream=filename)
    run.complete()

# %% [markdown]
# ## Go to Experiments Tab in the Azure ML Workspace Menu
# 

