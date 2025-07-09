# California Housing Price Prediction

A machine learning-powered pipeline to predict housing prices across California. This project features robust preprocessing, model tuning, evaluation, and learning curve visualization — all wrapped in a reproducible, Dockerized environment.

---

## Features

- Data cleaning, feature engineering, and log/ratio transformations
- Multiple regression models: SGD, Ridge, KNN, SVR, Decision Tree, Random Forest
- Hyperparameter tuning with `GridSearchCV` & `RandomizedSearchCV`
- Learning curves saved to `plots/` for model performance insight
- Docker support for portability and deployment

---

## Local Usage

```bash
#Clone the repo

git clone https://github.com/Im-Fardin/California-Housing-Price-Prediction.git
cd California-Housing-Price-Prediction

#Install dependencies

pip install -r requirements.txt

#Run the training pipeline

python src/train.py

#Docker Workflow

docker build -t cali-housing .

docker run --rm -v ${PWD}/plots:/app/plots cali-housing
