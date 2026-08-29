#Load extracted token features and turn them into matrices.
import pickle
import numpy as np


class Dataset:

    def __init__(self, train_pkl, test_pkl):
        with open(train_pkl, "rb") as f:
            self.train = pickle.load(f)
        with open(test_pkl, "rb") as f:
            self.test = pickle.load(f)

        self.feature_names = list(self.train["feature_names"])
        self.X_train = self._to_matrix(self.train)
        self.X_test = self._to_matrix(self.test)
        self.y_train = np.array(self.train["labels"])
        self.y_test = np.array(self.test["labels"])

        # Infinities come from ratio features such as halluc_risk_ratio.
        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=1e10, neginf=-1e10)
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=1e10, neginf=-1e10)

        n_neg = np.sum(self.y_train == 0)
        n_pos = np.sum(self.y_train == 1)
        self.scale_pos_weight = n_neg / n_pos

    def _to_matrix(self, split):
        return np.array([
            [row.get(name, 0.0) for name in self.feature_names]
            for row in split["features"]
        ])

    def summary(self):
        pos = np.sum(self.y_train == 1)
        return (f"{self.X_train.shape[1]} features | "
                f"train {len(self.y_train)} ({pos} positive, "
                f"{100 * pos / len(self.y_train):.2f}%) | "
                f"test {len(self.y_test)} | "
                f"scale_pos_weight {self.scale_pos_weight:.2f}")