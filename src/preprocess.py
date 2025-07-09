
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return df


def get_feature_pipeline() -> ColumnTransformer:
    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(_, feature_names_in):
        return ["ratio"]

    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    log_features = make_pipeline(
        KNNImputer(n_neighbors=5),
        FunctionTransformer(np.log1p, feature_names_out='one-to-one')
    )

    standard = make_pipeline(
        KNNImputer(n_neighbors=5),
        StandardScaler()
    )

    ratio = lambda cols: make_pipeline(
        KNNImputer(n_neighbors=5),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

    return ColumnTransformer([
        ("cat", categorical, ["ocean_proximity"]),
        ("log", log_features, ["total_bedrooms", "total_rooms", "latitude", "median_income"]),
        ("std", standard, ["housing_median_age", "longitude"]),
        ("ratio1", ratio(["total_bedrooms", "total_rooms"]), ["total_bedrooms", "total_rooms"]),
        ("ratio2", ratio(["total_rooms", "households"]), ["total_rooms", "households"]),
        ("ratio3", ratio(["median_income", "population"]), ["median_income", "population"])
    ], remainder="drop")
