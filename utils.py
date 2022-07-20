import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import hstack
from catboost import CatBoostRegressor, Pool

nltk.download("stopwords")


class RBKPreprocessor(BaseEstimator):
    def __init__(self):
        self._vectorizer_tags = TfidfVectorizer()
        self._vectorizer_authors = TfidfVectorizer()
        self._category_ohe = OneHotEncoder()
        self._category_from_title_ohe = OneHotEncoder()
        self._stop_words = stopwords.words("russian")
        self._stemmer = SnowballStemmer("russian")
        self._vectorizer_title = TfidfVectorizer()

    @staticmethod
    def _clean_list(title):
        return (title.
                replace("[", "").
                replace("]", "").
                replace(".", "").
                replace("'", "").
                replace(",", " ")
                )

    def _clean_title(self, title):
        if title.find("\n") > 0:
            title = title[0:title.find("\n\n")].lower()
        title = " ".join([self._stemmer.stem(w) for w in title.split() if w not in self._stop_words])
        return title

    def _find_category_in_title(self, title):
        if title.find("\n") > 0:
            title = title[title.find("\n\n"):].lower().strip()
        else:
            title = ""
        if "," in title:
            title = title[0:title.index(",")]
        else:
            title = ""

        return title

    def fit(self, df):
        self._category_ohe.fit(df.category.values.reshape(-1, 1))

        authors_clean = df.authors.apply(self._clean_list)
        tags_clean = df.tags.apply(self._clean_list)
        self._vectorizer_tags.fit(tags_clean)
        self._vectorizer_authors.fit(authors_clean)

        title_clean = df.title.apply(self._clean_title)
        category_from_title = df.title.apply(self._find_category_in_title)
        self._category_from_title_ohe.fit(category_from_title.values.reshape(-1, 1))
        self._vectorizer_title.fit(title_clean)

        return self

    def transform(self, df):
        ctr_zero = (df.ctr == 0)
        ctr_log = np.log(df.ctr)
        mean_ctr_log = np.mean(ctr_log.values, where=(ctr_log != -np.inf))
        ctr_log = np.where(df["ctr"] == 0, mean_ctr_log, ctr_log)

        category_sparse = self._category_ohe.transform(df.category.values.reshape(-1, 1))

        authors_clean = df.authors.apply(self._clean_list)
        tags_clean = df.tags.apply(self._clean_list)

        authors_count = authors_clean.apply(lambda x: len(x.split()))
        tags_count = tags_clean.apply(lambda x: len(x.split()))

        tags_sparse = self._vectorizer_tags.transform(tags_clean)
        authors_sparse = self._vectorizer_authors.transform(authors_clean)

        publish_date = pd.to_datetime(df.publish_date)
        publish_year = publish_date.dt.year * 100 + publish_date.dt.month
        publish_day = publish_date.dt.day
        publish_weekday = publish_date.dt.weekday
        publish_hour = publish_date.dt.hour

        title_clean = df.title.apply(self._clean_title)
        title_sparse = self._vectorizer_title.transform(title_clean)

        category_from_title = df.title.apply(self._find_category_in_title)
        category_from_title_sparse = self._category_from_title_ohe.transform(category_from_title.values.reshape(-1, 1))

        return hstack([
            ctr_zero.values[:, None],
            ctr_log[:, None],
            category_sparse,
            authors_count.values[:, None],
            tags_count.values[:, None],
            tags_sparse,
            authors_sparse,
            publish_year.values[:, None],
            publish_day.values[:, None],
            publish_weekday.values[:, None],
            publish_hour.values[:, None],
            title_sparse,
            category_from_title_sparse
        ])


class RBKRegressor(BaseEstimator):
    def __init__(self, n_iterations=100, random_seed=44):
        self.model1 = CatBoostRegressor(iterations=n_iterations,
                                        early_stopping_rounds=50,
                                        random_seed=random_seed,
                                        depth=8,
                                        learning_rate=0.1,
                                        eval_metric="R2",
                                        loss_function="RMSE")
        self.model2 = CatBoostRegressor(iterations=n_iterations,
                                        early_stopping_rounds=50,
                                        random_seed=random_seed,
                                        depth=10,
                                        learning_rate=0.15,
                                        eval_metric="R2",
                                        loss_function="RMSE")
        self.model3 = CatBoostRegressor(iterations=n_iterations,
                                        early_stopping_rounds=50,
                                        random_seed=random_seed,
                                        depth=10,
                                        learning_rate=0.15,
                                        eval_metric="R2",
                                        loss_function="RMSE")

    def fit(self, X_train, y_train, X_test, y_test):
        GOAL_NUM = 0
        self.model1.fit(X_train,
                        y_train.iloc[:, GOAL_NUM],
                        eval_set=Pool(
                            data=X_test,
                            label=y_test.iloc[:, GOAL_NUM])
                        )
        GOAL_NUM = 1
        self.model2.fit(X_train,
                        y_train.iloc[:, GOAL_NUM],
                        eval_set=Pool(
                            data=X_test,
                            label=y_test.iloc[:, GOAL_NUM])
                        )
        GOAL_NUM = 2
        self.model3.fit(X_train,
                        y_train.iloc[:, GOAL_NUM],
                        eval_set=Pool(
                            data=X_test,
                            label=y_test.iloc[:, GOAL_NUM])
                        )


def predict(self, X_test):
    predictions = np.zeros((X_test.shape[0], 3))
    predictions[:, 0] = self.model1.predict(X_test)
    predictions[:, 1] = self.model2.predict(X_test)
    predictions[:, 2] = self.model3.predict(X_test)
    return predictions


def get_mean_predictions(models, X_test):
    predictions = np.zeros((X_test.shape[0], 3))
    for model in models:
        predictions += model.predict(X_test)
    predictions /= len(models)
    return predictions


def local_metric(y_real, predictions, detail=True):
    overall = (
            0.4 * r2_score(y_real.iloc[:, 0], predictions[:, 0]) +
            0.3 * r2_score(y_real.iloc[:, 1], predictions[:, 1]) +
            0.3 * r2_score(y_real.iloc[:, 2], predictions[:, 2])
    )

    if not detail:
        return overall
    else:
        return (
            overall,
            0.4 * r2_score(y_real.iloc[:, 0], predictions[:, 0]),
            0.3 * r2_score(y_real.iloc[:, 1], predictions[:, 1]),
            0.3 * r2_score(y_real.iloc[:, 2], predictions[:, 2])
        )


def write_predictions(predictions, output_file="output.csv"):
    solution = pd.read_csv("data\sample_solution.csv")
    solution.iloc[:, 1:4] = predictions
    solution.to_csv("predictions/" + output_file, index=False)
