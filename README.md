"#Assignment 3 Prediction App" and FastAPI

Source: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

Project Description

A container is used to build and run the Prediction app, an app which predicts input user review text as positive
or negative. A dedicated API backend has been created and customized to wrap this backend with Docker to prepare for deployment and pushed to a GitHub repository.

Prerequisites

Docker must be installed to run app.

How to Run

Makefile is used to build and run the application with mapped commands for ease of use.
Use a browser to visit http://localhost:8501 to view application.


How to Clone Repositoy using GitHub
- Open main page of repository on GitHub
- Click on the "<>Code" button
-Copy the repository URL and select HTTPS, SSH key, or GitHub CLI
- Open Git Bash
- Change working directory to desired cloned directory location
- Use "git clone" command followed by the URL
- Press Enter to create clone.

API can be run and tested using Postman Desktop locally. 
API includes 4 distinct endopints:
- Health Check (get): confirms that the API is running
- Predict sentiment (post): takes text & predicts sentiment
- Probability (post): takes input, outputs sentiment & proba
- training example (get): returns random dataset for testing

