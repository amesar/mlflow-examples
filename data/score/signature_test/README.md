# Signature test scoring files

Files to test Model signatures for scoring. See [Model Signature](https://www.mlflow.org/docs/latest/models.html#model-signature) documentation.

There is a csv and json version in the respective subdirectory.

* wine-quality-white.csv|json - Normal file
* wine-quality-white-type.csv|json - One cell contains a string instead of a float
* wine-quality-white-less-columns.csv|json - One column is missing
* wine-quality-white-more-columns.csv|json - There exists an extra column
