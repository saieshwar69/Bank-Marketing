# Term Deposit Prediction: End-to-End Machine Learning project with MLOps


This project tackles the challenge of predicting bank deposit subscriptions using a comprehensive Machine Learning pipeline. It leverages the popular "Bank Marketing" dataset to train and deploy a model that optimizes F1-score and important performance metrics due to the dataset's imbalanced nature.

The project demonstrates a complete workflow, including data ingestion with CockroachDB, data exploration and feature engineering, model selection with hyperparameter tuning via MLFlow, and deployment to Microsoft Azure cloud using Docker containers and Github Actions for CI/CD. Additionally, a web application is built with PyWebIO and Flask for user interaction with the prediction model.

## Tech Stack

**Data Storage & Retrieval**
- *CockroachDB (SQL Database):* Efficiently stored and retrieved the "Bank Marketing" dataset for data ingestion..

**Data Preprocessing & Analysis** 
- *Python Libraries (Pandas, NumPy, Seaborn, etc.)*: Handled data manipulation, cleaning, exploration, and visualization, providing insights for feature engineering.

**Model Selection & Model Training** 
- *Machine Learning Algorithms (CatBoost, Random Forests, SVM, etc)*: Built, trained, and evaluated various machine learning models to predict bank deposit subscriptions.
- *MLFlow*: Tracked model experiments, hyperparameter tuning results, and other metrics (AUC-ROC, Precision, Recall) for optimal model selection.
- *DagsHub*: Remote server for comparing different experiments tracked by MLFlow.

**Web Application Development**
- *Docker*: Containerized the prediction pipeline, ensuring consistent runtime environment across deployment stages.
- *PyWebIO & Flask*: Developed an interactive web interface for users to leverage the trained model's predictions.

**Model Deployment** 
- *Microsoft Container Registry (MCR)*: Securely stored the Docker image for deployment.
- *Microsoft Azure Web App Service*: Hosted the web application built for user interaction with the prediction model.

**CI/CD (Continuous Integration/Continuous Delivery)**
- *GitHub Actions*: Automated the build, testing, and deployment pipeline for efficient and reliable updates.


## Selected Model's Performance

| Model | Accuracy    | F1-Score    |    ROC-AUC Score    | Precision | Recall |
| :---:   | :---: | :---: | :--: | :--: | :--: |
| CatBoostClassifier(iterations=600, depth=5, learning_rate=0.085) | 0.88   | 0.55   |  0.90    |   0.50   |   0.61   |

## Screenshots
- Web App Demo

<img src="https://i.ibb.co/5jNcRQg/ezgif-7-4e9458a4b4.gif" width="650">

- CockroachDB Interface

<img src="https://i.ibb.co/m84T9Xs/Screenshot-2024-03-26-223725.png" width="650">

- DagsHub MLFlow UI

<img src="https://i.ibb.co/7n9C1W8/Screenshot-2024-03-26-223540.png" width="650">

## Run Locally

Clone the project

```bash
  git clone https://github.com/rauhanahmed/Bank-Marketing
```

Go to the project directory

```bash
  cd Bank-Marketing
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python app.py
```

After performing the above steps, open any browser, and hit the localhost at port 80.


## Authors

[Rauhan Ahmed Siddiqui](https://linkedin.com/in/rauhan-ahmed/)


## License

[MIT](https://choosealicense.com/licenses/mit/)

