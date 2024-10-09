##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

# Updated by Mohan Vic on 06-10-2024 !not ready!

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import copy
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier


class ClassificationAlgorithms1:
    # Existing methods...

    # Apply a Logistic Regression approach for classification
    def logistic_regression(
        self, train_X, train_y, test_X, gridsearch=True, print_model_details=False
    ):
        from sklearn.linear_model import LogisticRegression

        if gridsearch:
            tuned_parameters = [{"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}]
            lr = GridSearchCV(
                LogisticRegression(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            lr = LogisticRegression()

        lr.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(lr.best_params_)

        if gridsearch:
            lr = lr.best_estimator_

        pred_prob_training_y = lr.predict_proba(train_X)
        pred_prob_test_y = lr.predict_proba(test_X)
        pred_training_y = lr.predict(train_X)
        pred_test_y = lr.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=lr.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=lr.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply an AdaBoost approach for classification
    def ada_boost(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=50,
        gridsearch=True,
        print_model_details=False,
    ):
        if gridsearch:
            tuned_parameters = [{"n_estimators": [50, 100, 200]}]
            ada = GridSearchCV(
                AdaBoostClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            ada = AdaBoostClassifier(n_estimators=n_estimators)

        ada.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(ada.best_params_)

        if gridsearch:
            ada = ada.best_estimator_

        pred_prob_training_y = ada.predict_proba(train_X)
        pred_prob_test_y = ada.predict_proba(test_X)
        pred_training_y = ada.predict(train_X)
        pred_test_y = ada.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=ada.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=ada.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a Voting Classifier for classification
    def voting_classifier(
        self, train_X, train_y, test_X, estimators, voting="soft", gridsearch=True
    ):
        if gridsearch:
            voting_clf = VotingClassifier(estimators=estimators, voting=voting)
            tuned_parameters = [{"voting": ["soft", "hard"]}]
            vc = GridSearchCV(voting_clf, tuned_parameters, cv=5, scoring="accuracy")
        else:
            vc = VotingClassifier(estimators=estimators, voting=voting)

        vc.fit(train_X, train_y.values.ravel())

        pred_prob_training_y = vc.predict_proba(train_X)
        pred_prob_test_y = vc.predict_proba(test_X)
        pred_training_y = vc.predict(train_X)
        pred_test_y = vc.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=vc.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=vc.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a Stacking Classifier for classification
    def stacking_classifier(
        self,
        train_X,
        train_y,
        test_X,
        base_estimators,
        final_estimator,
        gridsearch=True,
    ):
        if gridsearch:
            stacking_clf = StackingClassifier(
                estimators=base_estimators, final_estimator=final_estimator
            )
            tuned_parameters = [{}]  # Define appropriate params if needed
            sc = GridSearchCV(stacking_clf, tuned_parameters, cv=5, scoring="accuracy")
        else:
            sc = StackingClassifier(
                estimators=base_estimators, final_estimator=final_estimator
            )

        sc.fit(train_X, train_y.values.ravel())

        pred_prob_training_y = sc.predict_proba(train_X)
        pred_prob_test_y = sc.predict_proba(test_X)
        pred_training_y = sc.predict(train_X)
        pred_test_y = sc.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=sc.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=sc.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply XGBoost
    def xgboost_classifier(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        gridsearch=True,
    ):
        if gridsearch:
            tuned_parameters = [
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                }
            ]
            xgb = GridSearchCV(
                XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                tuned_parameters,
                cv=5,
                scoring="accuracy",
            )
        else:
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                use_label_encoder=False,
                eval_metric="logloss",
            )

        xgb.fit(train_X, train_y.values.ravel())

        pred_prob_training_y = xgb.predict_proba(train_X)
        pred_prob_test_y = xgb.predict_proba(test_X)
        pred_training_y = xgb.predict(train_X)
        pred_test_y = xgb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=xgb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=xgb.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply LightGBM
    def lightgbm_classifier(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=100,
        max_depth=-1,
        learning_rate=0.1,
        gridsearch=True,
    ):
        if gridsearch:
            tuned_parameters = [
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [-1, 3, 5],
                    "learning_rate": [0.01, 0.1, 0.2],
                }
            ]
            lgb_model = GridSearchCV(
                lgb.LGBMClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )

        lgb_model.fit(train_X, train_y.values.ravel())

        pred_prob_training_y = lgb_model.predict_proba(train_X)
        pred_prob_test_y = lgb_model.predict_proba(test_X)
        pred_training_y = lgb_model.predict(train_X)
        pred_test_y = lgb_model.predict(test_X)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=lgb_model.classes_
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=lgb_model.classes_)

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply CatBoost
    def catboost_classifier(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=100,
        depth=6,
        learning_rate=0.1,
        gridsearch=True,
    ):
        if gridsearch:
            tuned_parameters = [
                {
                    "n_estimators": [50, 100, 200],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.1, 0.2],
                }
            ]
            catboost_model = GridSearchCV(
                CatBoostClassifier(silent=True),
                tuned_parameters,
                cv=5,
                scoring="accuracy",
            )
        else:
            catboost_model = CatBoostClassifier(
                n_estimators=n_estimators,
                depth=depth,
                learning_rate=learning_rate,
                silent=True,
            )

        catboost_model.fit(train_X, train_y.values.ravel())

        pred_prob_training_y = catboost_model.predict_proba(train_X)
        pred_prob_test_y = catboost_model.predict_proba(test_X)
        pred_training_y = catboost_model.predict(train_X)
        pred_test_y = catboost_model.predict(test_X)
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=catboost_model.classes_
        )
        frame_prob_test_y = pd.DataFrame(
            pred_prob_test_y, columns=catboost_model.classes_
        )

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Add any other existing methods...


# Example of how to use the new class (commented out for your implementation)
# classifier = ClassificationAlgorithms()
# train_X, train_y, test_X = ... # Load your data here
# classifier.logistic_regression(train_X, train_y, test_X)
# classifier.ada_boost(train_X, train_y, test_X)
# classifier.voting_classifier(train_X, train_y, test_X, estimators=[('rf', RandomForestClassifier()), ('svc', SVC(probability=True))])
# classifier.stacking_classifier(train_X, train_y, test_X, base_estimators=[('rf', RandomForestClassifier()), ('svc', SVC(probability=True))], final_estimator=LogisticRegression())
# classifier.xgboost_classifier(train_X, train_y, test_X)
# classifier.lightgbm_classifier(train_X, train_y, test_X)
# classifier.catboost_classifier(train_X, train_y, test_X)
