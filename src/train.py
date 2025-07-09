import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
)
from sklearn.linear_model import SGDRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from preprocess import load_data, get_feature_pipeline

# Ensure output folder exists
os.makedirs("plots", exist_ok=True)


def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 40)
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training RMSE", color="blue", marker="o")
    plt.plot(train_sizes, test_scores_mean, label="CV RMSE", color="orange", marker="o")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    filename = title.lower().replace(" ", "_").replace("-", "").replace("__", "_")
    plt.savefig(f"plots/{filename}.png")
    plt.close()


def main():
    # Load and split data
    df = load_data("data/housing.csv")
    X = df.drop(columns=["median_house_value", "income_cat"])
    y = df["median_house_value"]
    strat_col = df["income_cat"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=strat_col, random_state=42
    )

    preprocessor = get_feature_pipeline()

    models = [
        ("SGD", SGDRegressor(random_state=42), {
            "sgdregressor__alpha": [1e-5, 1e-4, 1e-3],
            "sgdregressor__max_iter": [1000, 2000]
        }, GridSearchCV),
        ("Ridge", Ridge(), {}, RidgeCV),
        ("KNN", KNeighborsRegressor(), {
            "kneighborsregressor__n_neighbors": [5, 10],
            "kneighborsregressor__weights": ["uniform", "distance"],
            "kneighborsregressor__metric": ["euclidean", "manhattan"]
        }, GridSearchCV),
        ("SVR", SVR(), {
            "svr__C": [10, 100],
            "svr__epsilon": [0.1, 0.5],
            "svr__kernel": ["rbf"],
            "svr__gamma": ["scale", "auto"]
        }, RandomizedSearchCV),
        ("Tree", DecisionTreeRegressor(random_state=42), {
            "decisiontreeregressor__max_depth": [10, 20],
            "decisiontreeregressor__min_samples_split": [2, 5],
            "decisiontreeregressor__min_samples_leaf": [1, 2]
        }, RandomizedSearchCV),
        ("RF", RandomForestRegressor(random_state=42), {
            "randomforestregressor__n_estimators": [50, 100],
            "randomforestregressor__max_depth": [10, 20],
            "randomforestregressor__min_samples_split": [2, 5],
            "randomforestregressor__min_samples_leaf": [1, 2]
        }, RandomizedSearchCV),
    ]

    for name, regressor, param_grid, search_cls in models:
        print(f"\nTraining: {name}")
        pipe = make_pipeline(preprocessor, regressor)

        if search_cls is RidgeCV:
            search = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100],
                             cv=3, scoring="neg_root_mean_squared_error")
            search.fit(preprocessor.fit_transform(X_train), y_train)
            print(f"Best alpha: {search.alpha_}")
            pipe.set_params(ridge__alpha=search.alpha_)

        else:
            if search_cls is RandomizedSearchCV:
                search = search_cls(
                    pipe,
                    param_grid,
                    n_iter=10,
                    cv=3,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    verbose=1
                )
            else:
                search = search_cls(
                    pipe,
                    param_grid,
                    cv=3,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    verbose=1
                )
            search.fit(X_train, y_train)
            print(f"Best score: {-search.best_score_:.2f}")
            print(f"Best params: {search.best_params_}")
            pipe.set_params(**search.best_params_)

        # Plot learning curve
        plot_learning_curve(pipe, X_train, y_train, title=f"Learning Curve for {name}")


if __name__ == "__main__":
    main()
