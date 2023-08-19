# Databricks notebook source
# MAGIC %md ## Speech Recognition Task
# MAGIC
# MAGIC ##### Hugging Face model 
# MAGIC * Task: automatic-speech-recognition
# MAGIC * Model: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
# MAGIC * Description: automatic speech recognition (ASR) and speech translation
# MAGIC
# MAGIC ##### MLflow `transformers` flavor
# MAGIC ```
# MAGIC {
# MAGIC   "transformers_version": "4.28.1",
# MAGIC   "code": null,
# MAGIC   "task": "automatic-speech-recognition",
# MAGIC   "instance_type": "AutomaticSpeechRecognitionPipeline",
# MAGIC   "source_model_name": "openai/whisper-tiny",
# MAGIC   "pipeline_model_type": "WhisperForConditionalGeneration",
# MAGIC   "framework": "pt",
# MAGIC   "feature_extractor_type": "WhisperFeatureExtractor",
# MAGIC   "tokenizer_type": "WhisperTokenizer",
# MAGIC   "components": [
# MAGIC     "feature_extractor",
# MAGIC     "tokenizer"
# MAGIC   ],
# MAGIC   "model_binary": "model"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC #### Based upon
# MAGIC * [github.com/mlflow/mlflow/examples/transformers/sentence_transformer.py](https://github.com/mlflow/mlflow/blob/master/examples/transformers/sentence_transformer.py)

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
registered_model_name = dbutils.widgets.get("1. Registered model")
registered_model_name

# COMMAND ----------

client = get_client(registered_model_name)

# COMMAND ----------

# MAGIC %md ### Setup transformer

# COMMAND ----------

import transformers
from packaging.version import Version
import requests

import mlflow

# Acquire an audio file
audio = requests.get("https://www.nasa.gov/62283main_landing.wav").content

task = "automatic-speech-recognition"
architecture = "openai/whisper-tiny"

model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
audio_transcription_pipeline = transformers.pipeline(
    task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that if the input type is of raw binary audio, the generated signature will match the
# MAGIC one created here. For other supported types (i.e., numpy array of float32 with the
# MAGIC correct bitrate extraction), a signature is required to override the default of "binary" input
# MAGIC type.

# COMMAND ----------

signature = mlflow.models.infer_signature(
    audio,
    mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio),
)

inference_config = {
    "return_timestamps": "word",
    "chunk_length_s": 20,
    "stride_length_s": [5, 3],
}

# COMMAND ----------

# MAGIC %md ### Log Model

# COMMAND ----------

# Log the pipeline
with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=audio_transcription_pipeline,
        artifact_path="whisper_transcriber",
        signature=signature,
        input_example=audio,
        inference_config=inference_config,
    )

# COMMAND ----------

# MAGIC %md ### Register model

# COMMAND ----------

version = register_model(client, registered_model_name, model_info, run)

# COMMAND ----------

dump_obj(version)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

# Load the pipeline in its native format
loaded_transcriber = mlflow.transformers.load_model(model_uri=model_info.model_uri)

transcription = loaded_transcriber(audio, **inference_config)

print(f"\nWhisper native output transcription:\n{transcription}")

# Load the pipeline as a pyfunc with the audio file being encoded as base64
pyfunc_transcriber = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

pyfunc_transcription = pyfunc_transcriber.predict([audio])

# Note: the pyfunc return type if `return_timestamps` is set is a JSON encoded string.

print(f"\nPyfunc output transcription:\n{pyfunc_transcription}")

# COMMAND ----------

# MAGIC %md ### Return

# COMMAND ----------

dbutils.notebook.exit(create_results(model_info, version))
