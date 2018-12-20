# Project Recommender System

In this project, we study the development of a straightforward recommendation system, based on a training set containing 10000 users and 1000 movies. In particular our project aims to predict the rating these users would give to new movies, and new users would give to these movies. Our project does not address temporal changes in the dataset or the "Cold Start" problem -where new users with no reviews are added to the dataset - directly.

In the course of our investigation we developed 17 models. Drawing inspiration from the results of the Netflix Challenge, we developed a blended model, which depends on these 17 models, and complements their respective strengths and weaknesses, creating a more accurate prediction.

## Dependencies

* Install a python environment.

* Install requirements for this project
	* Run: pip install -r <folder_of_project>/requirements.txt  


* Data Sets:
    * Download the data_train.csv and sampleSubmission.csv files from the competition page on CrowdAI. Please rename sampleSub
	  (https://www.crowdai.org/challenges/epfl-ml-recommender-system/dataset_files). The files should be put in the folder data.


## Description of files

*
