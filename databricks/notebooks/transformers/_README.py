# Databricks notebook source
# MAGIC %md ### Hugging Face Transformers
# MAGIC
# MAGIC ##### Overview
# MAGIC * Log various Hugging Face models for different tasks as MLflow runs and register them as a model.
# MAGIC * The `transformers` flavor attributes are logged as run and model versions tags (staring with `hf_`). See [example](https://github.com/amesar/mlflow-reports/blob/master/samples/databricks/model_reports/transformers/Conversational_Task/report.json#L469).
# MAGIC * Sample UC and non-UC models: [Conversational_Task](https://github.com/amesar/mlflow-reports/blob/master/samples/databricks/model_reports/transformers/Conversational_Task) - JSON and markdown report.
# MAGIC
# MAGIC ##### Transformer model notebooks
# MAGIC
# MAGIC | Notebook | Hugging Face Task | Hugging Face Model | Model GB|
# MAGIC |-----|-----|-------|---|
# MAGIC | [Text_to_Text_Generation_Task]($Text_to_Text_Generation_Task) | [text2text-generation](https://huggingface.co/tasks/text-generation) | [declare-lab/flan-alpaca-base](https://huggingface.co/declare-lab/flan-alpaca-base) | 1.092 |
# MAGIC | [Translation_Task]($Translation_Task) | [translation_en_to_fr](https://huggingface.co/tasks/translation) | [t5-small](https://huggingface.co/t5-small) |  0.245 |
# MAGIC | [Conversational_Task]($Conversational_Task) | [conversational](https://huggingface.co/tasks/conversational) | [microsoft/DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) | 1.654 |
# MAGIC | [Feature_Extraction_Task]($Feature_Extraction_Task) | [feature-extraction](https://huggingface.co/tasks/feature-extraction) | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 0.134 |
# MAGIC | [Speech_Recognition_Task]($Speech_Recognition_Task) | [automatic-speech-recognition](https://huggingface.co/tasks/automatic-speech-recognition) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |  0.154 |
# MAGIC
# MAGIC ##### Other notebooks
# MAGIC * [Run_Task_Notebooks]($Run_Task_Notebooks) - Run all transformer notebooks.
# MAGIC * [Common]($Common)
# MAGIC * [Template]($Template) - template for new transformer notebooks.
# MAGIC
# MAGIC ##### Originally from
# MAGIC * https://github.com/mlflow/mlflow/blob/master/examples/transformers
# MAGIC
# MAGIC Last updated: 2023-08-20
