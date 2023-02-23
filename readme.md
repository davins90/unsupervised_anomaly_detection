# 1) General Overview

This folder contains code useful for the development of the final [FourhBrain](https://fourthbrain.ai/courses/machine-learning-engineer/) MLE10 course project. 

As a project, the one of a company specializing in cashback was chosen. Given the particular operating environment, the risk of endorsing fraudulent transactions is very high. Equally risky for business profitability is the lost revenue generated when customers do not take full advantage of the opportunities presented.

With this purpose therefore, this project was implemented: 
- on the one hand, an engine capable of detecting fraudulent transactions was built;
- on the other, an engine capable of clustering customers so that offers to them could be better tailored.

A note on the work: since no initial dataset was provided that was deemed adequate for the purpose, a dataset found among those provided by [AWS in its folder](https://github.com/amazon-science/fraud-dataset-benchmark) was used for development. In this way, what is intended to emerge with this work is more than an already operational solution for the client, a **methodological approach** that can be adapt and make work on an official database.

The following are the links to the deployed solutions:
    - [frontend application](https://frontend-4b-ylpi3mxsaq-oc.a.run.app/)
    - [backend API endpoints](https://backend-4b-ylpi3mxsaq-oc.a.run.app/docs)

# 2) Tech environment
Technologically speaking, the development of the project can be divided into two phases.
- The first, development, in which three docker containers managed by a single orchestrator ([docker-compose](https://docs.docker.com/compose/)) were used. Below are the specifications of the three containers:
    - container environment_4b: in which, on port 8080, the jupyter lab environment useful for generating the code needed for the modeling and pre-processing phase is generated;
    - container streamlit: in which, on port 8501, the code useful for the frontend part of the web application was generated, by using [Streamlit](https://streamlit.io/) framework;
    - container fastapi: in which, on port 8000, the code useful for the querying part of the APIs specifically created and queried by the forntend to generate the results was generated. [FAST API](https://fastapi.tiangolo.com/) was selected as backend API library.

- The second, production part,  was built with the environment in which this part would live, i.e., Google Cloud Product, in mind. For the production deployment of the two main parts (backend and frontend), it was decided to generate two separate containers: 
    - the first, backend, with the available API endpoints at the [following page](https://backend-4b-ylpi3mxsaq-oc.a.run.app/docs); 
    - the second, frontend, with the web app built to query the APIs and navigate the data, at the [following page](https://frontend-4b-ylpi3mxsaq-oc.a.run.app/).

# 3) Modeling pipeline
- **Data ingestion**: in the context of this project, this step consisted of creating the dataset after retrieving it from the source reported by Amazon Science. Once retrieved it was filtered by some fields to make it as generalist as possible to the case at hand. This is a dataset of structured data (both categorical and continuous) in which there is a label on the transaction, if it is or it isn't a fraud.
- **EDA**: A distinction needs to be made for the exploration phase on the data:
    - continuous data have a strong asymmetry, which will be appropriate to deal with later with a log-transofrmation;
    - for categorical data on the other hand, which are the majority, through the factorial analysis offered by [MCA](https://en.wikipedia.org/wiki/Multiple_correspondence_analysis), it is possible to observe the relationships that exist between them and find out if already at this stage relationships and patterns emerge, useful for our purposes. 
    - Wanting to extend the analysis not only to the individual subjects under examination (the fraudulent transactions), but to the relationships between them, it was thought to construct a [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) useful for highlighting patterns based on the specially created relationships between the identified nodes.
- **Data preparation**: after obtaining the "fac-simile" dataset of an official dataset, the strong imbalance in the class being predicted became apparent. Three different datasets were generated before rebalancing: the one useful for training, validation phase and testing. To make sure that no distribution shift is present, the adversarial validation procedure was used to check the similarity between the two datasets. Having ascertained this step, undersampling of the majority class on the training set was performed to remedy the imbalance, a decision that in the trade-off between loss of information and addition of bias (with oversampling or artificial data) seemed best.
- **Features engineering**: to improve the predictive ability of the model, some features were artificially constructed and some categories of categorical features were built.
- **Modeling**: for this step it is necessary to do some in-depth study. 
    -  Regards to the fraud prediction, given the sensitive topic under consideration, it was thought that it might be useful to build a framework that can take into account two aspects:
        - the output, expressed in probabilistic terms, of a machine learning algorithm, an approach useful for detecting patterns not immediately recognizable thanks to the optimization and mathematical description processes of the problem.
        - the output, expressed in a range between 0 and 1, of a weighted average of certain features determined by the human experience of investigators

        Finally, the previous two scores are "merged" into a final score through a technique that exploits [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics). A useful approach to tacking on a framework that can update itself automatically, considering not only the estimated point, but the entire distribution of results.
    - Regarding the clustering engine used to identify Marketing Personas with which to then associate customers, the [k-prototypes algorithmic](https://github.com/nicodv/kmodes) solution was used, which is useful for working with both categorical and continuous data, as in the case of the dataset under consideration.

        When evaluating the model, [Gower's distance](https://statisticaloddsandends.wordpress.com/2021/02/23/what-is-gowers-distance/) was used to correctly calculate the silhouette metric to handle both categorical and continuous data. This metric infact on categorical data applies Dice's coefficient, a measure of similarity, while on continuous data it can use either Euclidean or Manhattan distance.
        
        The value generated by the metric on the four selected clusters was 0.21. Once the four cluster labels have been identified and the centroids described (as carried out within the web app), the saved model can be used to make predictions.

# 4) Focus on the Machine Learning Training Step
An image of the code written to perform model training is shown below. To avoid data leakage problems, it was decided to use [Sklearn's Pipeline class](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to perform the transformation operations on the training set.
Then with the [Grid Search class](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and a set of parameters from which to search for the best alternative, we obtain the final model to be used. The [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) was chosen as the algorithm, which is a robust, accurate solution and, due to its adaptive nature, is able to improve and correct itself during training. I had to generate as output the probability of belonging to a given class, it was necessary to perform calibration of the model used, so as to refine its prediction capabilities.

![image pipeline](https://github.com/davins90/unsupervised_anomaly_detection/blob/master/pipeline_sklearn.png)

Regarding this section, within the built web app, a page was implemented that was useful for performing model retraining based on other parameters entered as input.
This step was made possible through the [Ploomber Engine library](https://github.com/ploomber/ploomber-engine), through which it is possible from a python file to re-run jupyter notebooks and pass parameters that interact with notebooks. This is an alternative approach the the widly use Apache Airflow scheduler.

## 4.1) Metrics
Recall was chosen as the metric for model evaluation and optimization. This choice is dictated by the fact that the cost of false negatives, that is, of predicting as "not fraudulent" a transaction that actually is, is greater than the cost of false positives. 

![image confusion matrix](https://github.com/davins90/unsupervised_anomaly_detection/blob/master/confusion_matrix.png)

# 5) Web App Development
The development of the demo designed for the project, as written earlier, sees the existence of a backend (Fast Api) and a frontend (Streamlit). With regard to the backend, five POST endpoints were constructed that serve the following purposes:
- three related to obtaining score results related to fraud prediction (two working on a single user's data passed as input and one working on an entire dataset)
- two related to obtaining the label of the cluster of marketing personas to which the customer data (or customers if a dataset is passed) belongs.

On the front end, instead, the following pages were constructed:
- a homepage, with a brief introduction to the demo
- explorative analysis, a page where you can navigate through the dataset to extract descriptive stastics, analyze distributions, the generated graph, and perform factor analysis to discover latent structures
- fraud analysis, a page where it is possible to obtain the fraud probability score of a transaction according to different modes: single user, dataset, or perform model retraining
- customer personas, a page where you can get the label generated by the clustering model to associate with the customer or customers in a dataset
- documentation, a page where document references are present.

The operation/building block of the application is as follows:
- Database: Google Drive folder (moving to Cloud Storage asap).
- Code repository: GitHub. In case of code update, via the Cloud Build feature, CI/CD functionality has been implemented automatically, which uploads the updated code.
- Deploy: a serverless Docker container, Cloud Run, for final deployment (two cloud run container, for backend and frontend)

It is worth noting that the developed application is designed for two use cases:
- the fraud analysis team with regard to the fraud prediction engine
- the marketing team for the prediction engine of the marketing personas to which the customer belongs. Engine useful in triggering highly personalized call to action.

Further specifics can be found within the application and in a future flowchart.

# 6) Tech Stack and Reproducibility

To replicate this project locally, once you have downloaded the folder, with docker and docker-compose installed on your machine, simply run the following commands:
- docker compose build: to create the environment with the relevant libraries found in the requirements files in the packages folder;
- docker compose up: once the service is created you can launch the three containers to be displayed in the browser at the following ports on your localhost:
    - jupyter lab: 8080
    - streamlit: 8501
    - fast api: 8000/docs
- docker compose down: to tunr off the service

![Here the image of the diagram of the overall solution](https://github.com/davins90/unsupervised_anomaly_detection/blob/master/decision_intelligence_application_3.png)


# 7) Notes and future development
Future developments of this work may be:
- In terms of web apps:
    - improve usability and the ability to give input of other parameters
    - moving files and templates from Google Drive to Cloud Storage.
    - better integrate Ploomber.
    - Authetication layer for the fraud and marketing teams with this [Streamlit component](https://github.com/mkhorasani/Streamlit-Authenticator)
- In terms of modeling:
    - try naive Bayes as an algorithm, through a solution that can handle data of different types
    - HGB, as an alternative boosting algorithmù
    - Variational Auto Encoder and transformer
    - GNN to take full advantage of the constructed graph structure, at present only for exploratory purposes. 
    - enhancing clustering model with other approaches and other features by looking at the features importances from the classification model.
- Problems:
    - here two links for addressing problem on the visualizaion of matplotlib chart inside streamlit after deployment on GCP Cloud Run ([one](https://github.com/streamlit/streamlit/issues/1294), [two](https://cloud.google.com/run/docs/configuring/session-affinity?hl=it))

