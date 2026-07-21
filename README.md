# SENTIMENT ANALYSIS MLOPS

## Introduction

The following project tries to implement an automation pipeline for the deployment and retraining of a sentiment analysis model on huggingface.
The core idea was to create an huggingface space to serve the model, which would contain the main links for the creator of the space and an interface from which the model could be tested.  
Automated free reproducibility was a main goal -   
After registration in github, huggingface, kaggle and grafana and the set up of the variables and secrets as per env_example, a simple push to the main branch on a Git Repo of the complete project would have been sufficient to implement all the steps.  

**Recent changes in huggingface pricing during the setup of the project have however  caused docker based spaces like this to fall under PRO account category, which requires a monthly subscription, with only static pages allowed for free hosting.**

If you're an huggingface PRO subscriber, after setup of variables and secrets, removing comment-out of   
api.create_repo()  
line in -> scripts/deploy_space.py 
will allow a direct setup of the space page on huggingface as this one [Sentiment_Analysis](
https://divde-sentiment-proj-5.hf.space) 
The above listed space contains direct links to access all other deployed integrations of the app  
(Retrain, Data Generation, Grafana, MLFLOW, Datasets)
Note: config files kernel-metadata for kaggle defaults to private:false for public review on this project  
however during testing appeared that sometimes kaggle reverted the visibility to private.  
It can be manually set in the notebook settings, where it i indicated with a padlock.

Deployment of the application under the above circumstances would be also possible from a local airflow setup included in this repo or directly launching the deploy_space.py


| TECH | TASK | RESULT |
| --- | --- | --- |
| FASTAPI | Dynamics endpoints allowing rerouting<br />for predictions,model analysis with mlflow,<br />logs and monitoring stats collection  | endpoints<br />/predict <br /> /logs<br />/daily_stats<br />/weekly_stats<br />/mlflow/*|
| MLFLOW | Training and performance analysis<br />with model versioing history | Freely accessible MLflow URI <br />integrated in the application|
|GITHUB<br />ACTIONS|Orchestration and live <br />deployment of new main branch | Automated <br />-Collection of new post<br /> through scheduled script<br />-Drift check on monitoring endpoint to trigger retrain scheduled daily<br />-Grafana dashboard upload routed to <br />logs and data endpoints on the app<br />-Retrain of model through push of notebook and secrets private database on kaggle and deploy to HF repo on validation<br />-Synthetic data generation through small LLM fed with real post fetched from the data collection API scheduled weekly<br />-Deploy of variables and secrets to HF space for the application<br />-Deploy of HF Space in a docker space<br />-Application testing and validation in production   |
|GRAFANA|Monitoring and visual evaluation|Visual collection of performance for the model and the Company (Aspect)<br />under observation in the app and fetching script<br />Any modification required in the dashboards can be saved and exported in the classic code format,then saved in the sentiment.json file.<br />Additional formatting will occur during the upload workflow to ensure compatibility with the grafana cloud representation|
|TRANSFORMERS| Training, evaluation and inference|Peft-LoRA training pipeline with implementation of synthetically generated data and weak labelling for new real data|
|SQLITE|Storage of predictions,posts and statistics|Logs predictions, text confidence for the app to serve metrics endpoints,<br /> offer possibility to export in excel the data in the time range selected|
|AIRFLOW|Orchestration and Deployment|A dockerized instance of AIRFLOW is included in the repository as an alternative meaning of orchestration for the scheduled and operational pipelines, for local deployment or on VM indipendently of Github. Unit test integrated in the deploy pipeline while dockerized integration test remain as a git workflow only|
|MASTODON API|Free API used to collect data for this project|Mastodon is a niche social media which include an accessible API for fetching and posting data. Script included will filter english posts related to the specific Company hashtag and store them in a viewable json in the HF account with its  prediction from the online model|


## Model and Aspect Based Sentiment Analysis (ABSA)

The base model adopted for this project is a BERT base model  
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
trained on 124M tweets on various language tasks  
In this instance the model will be fed data collected automatically from the mastodon api regarding a specific hastag for a selected company or topic, from which a sentiment will be extracted.  
The posts and sentiments collected will then be stored in a sqlite DB on the app for review and evaluation.   
While reviewing data and predictions collected i noticed how in some occasions the general sentiment of the post was in contrast with the opinion on the company i was trying to evaluate. 
Exploring further I encountered the ABSA technique for aspect based analysis, where in the most common languages models trained for nlp task, it allows the model to focus on a specific aspect of the text in analysis.The specific ABSA task implemented in these experiments is the aspect term polarity, which aim to identify the sentiment expressed with regards to the specific aspect fed to the model. This implementation is present but not completed, as it has been simplified to a single ASPECT injected through variable in the App. Evaluation on generict aspect - text sentiment relationship is included in the training and evaluation, but real test evaluation for the model capacity to deeply associate the domain specific context to the object of the analysis is still unclassified.
On first start-up, the model will be not fine-tuned on ABSA specific classification, though it will accept the input text and ASPECT. After the first training it will start to proogressively adapt to the aspect based classification through the public training set and the generated datas provided.

## Results

All pipelines steps are functional and tested for both orchestrators (for airflow in local environment only and deployment to HF) once the variables and secrets are properly set.  
Performance-wise, as far as observable from retrieved post and classification in the dataset built from the api data collection script, the model is not yet behaving solidly.  
Mastodon posts are rather noisy even after clean up, and a single source shows limitations in the quality of the improvements for the model. 
A stronger LLM Judge or manual annotation for the posts would probably fare better, but due to infrastructure constraints and automated pipeline focus it falls behind the scope of the current project.  
Training shows good results for the model imporvements on aspect based classification from the split training dataset, though weak labelled real posts were not considered here as a valuable evaluation source


