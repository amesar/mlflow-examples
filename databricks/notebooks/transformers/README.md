
### Hugging Face Transformers

##### Overview
* Log various Hugging Face models for different tasks as MLflow runs and register them as a model.
* The `transformers` flavor attributes are logged as run and model versions tags (staring with `hf_`). See [example](https://github.com/amesar/mlflow-reports/blob/master/samples/databricks/model_reports/transformers/Conversational_Task/report.json#L469).
* Sample UC and non-UC models: [Conversational_Task](https://github.com/amesar/mlflow-reports/blob/master/samples/databricks/model_reports/transformers/Conversational_Task) - JSON and markdown report.

##### Transformer model notebooks

| Notebook | Hugging Face Task | Hugging Face Model | Model GB|
|-----|-----|-------|---|
| [Text_to_Text_Generation_Task](Text_to_Text_Generation_Task.py) | [text2text-generation](https://huggingface.co/tasks/text-generation) | [declare-lab/flan-alpaca-base](https://huggingface.co/declare-lab/flan-alpaca-base) | 1.092 |
| [Translation_Task](Translation_Task.py) | [translation_en_to_fr](https://huggingface.co/tasks/translation) | [t5-small](https://huggingface.co/t5-small) |  0.245 |
| [Conversational_Task](Conversational_Task.py) | [conversational](https://huggingface.co/tasks/conversational) | [microsoft/DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) | 1.654 |
| [Feature_Extraction_Task](Feature_Extraction_Task.py) | [feature-extraction](https://huggingface.co/tasks/feature-extraction) | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | 0.134 |
| [Speech_Recognition_Task](Speech_Recognition_Task.py) | [automatic-speech-recognition](https://huggingface.co/tasks/automatic-speech-recognition) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |  0.154 |

##### Other notebooks
* [Run_Task_Notebooks](Run_Task_Notebooks.py) - Run all transformer notebooks.
* [Common](Common.py)
* [Template](Template.py) - template for new transformer notebooks.

##### Originally from
* https://github.com/mlflow/mlflow/blob/master/examples/transformers

Last updated: 2023-12-12
