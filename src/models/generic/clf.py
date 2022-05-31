# Global import
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from typing import List, Union, Dict
from matplotlib import pyplot as plt
import xgboost 

# local import / global var


   
class LRClassifier:
    """Classifier (Logistic Regression)."""
    
    def __init__(
        self, lr_params: Dict[str, Union[str, int, float]], feature_names: List[str]
    ):
        """Wrapper of Scikit learn logistic regression model for binary classification.

        Args:
            lr_params (Dict[str, Union[str, int, float]]): parameters of scikit-learn model.
            feature_names (List[str]): List of name of features.
            categorical_features (List[str]): List of name of categorical features.
        """
        self.feature_names = feature_names
        self.clf = LogisticRegression(**lr_params)
        self.is_fitted = False
            
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Classifier not fitted yet.')
        y_hat = self.clf.predict(X)
        return y_hat
    
    def predict_proba(self, X):
        if not self.is_fitted:
            
            raise ValueError('Classifier not fitted yet.')
        
        y_hat = self.clf.predict_proba(X)
        
        return y_hat
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        y_hat_train, y_hat_test = self.predict(X_train), self.predict(X_test)
        return {
            'train_accuracy': accuracy_score(y_train, y_hat_train), 
            'test_accuracy': accuracy_score(y_test, y_hat_test), 
            'train_cm': confusion_matrix(y_train, y_hat_train), 
            'test_cm': confusion_matrix(y_test, y_hat_test), 
        }
    
    def plot_feature_importance(self, max_num_features=None, figsize=None):
        pass
                
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)   
    
    
class XgboostClassifier:
    """Classifier (XGBoost classifier)."""
    
    def __init__(
        self, xgb_params: Dict[str, Union[str, int, float]], feature_names: List[str]
    ):
        """Wrapper of xgboost model for binary classification.

        Args:
            xgb_params (Dict[str, Union[str, int, float]]): Xgboost parameters.
            feature_names (List[str]): List of name of features.
            categorical_features (List[str]): List of name of categorical features.
        """
        self.feature_names = feature_names
        self.clf = xgboost.XGBClassifier(use_label_encoder=False, **xgb_params)
        self.is_fitted = False
            
    def fit(self, X, y):
        self.clf.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError('Classifier not fitted yet.')
        y_hat = self.clf.predict(X, validate_features=False)
        return y_hat
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError('Classifier not fitted yet.')
        
        y_hat = self.clf.predict_proba(X, validate_features=False)
        
        return y_hat
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        y_hat_train, y_hat_test = self.predict(X_train), self.predict(X_test)
        return {
            'train_accuracy': accuracy_score(y_train, y_hat_train), 
            'test_accuracy': accuracy_score(y_test, y_hat_test), 
            'train_cm': confusion_matrix(y_train, y_hat_train), 
            'test_cm': confusion_matrix(y_test, y_hat_test), 
        }
    
    def plot_feature_importance(self, max_num_features=None, figsize=None):
        
        # Create figure
        plt.figure(figsize=figsize)
        ax = plt.subplot()
        
        self.clf.get_booster().feature_names = self.feature_names
        xgboost.plot_importance(self.clf.get_booster(), ax=ax, max_num_features=max_num_features)
        plt.show()
                
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
