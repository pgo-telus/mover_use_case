# Global import
from telnetlib import IP
from sklearn.model_selection import train_test_split
from typing import List, Any, Dict, Tuple
from yaml import safe_load
from pathlib import Path
import numpy as np
import itertools
import pickle
import json
import sys
import os

# Local imports and params
pth_project = Path(__file__.split('core')[0])
pth_training_data = pth_project / 'data' / 'training'
pth_models = pth_project / 'data' / 'models'
pth_parameters = pth_project / 'core' / 'parameters' / 'movers.yaml'
sys.path.insert(0, str(pth_project))

from core.models.generic.clf import LRClassifier as MoverClassifier


def fit(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    lr_params: Dict[str, Any], lr_grid_params: Dict[str, List[Any]], 
    l_feature_names: List[str], do_grid_search: bool = True
) -> Tuple[MoverClassifier, Dict[str, Any]]:
    
    if not do_grid_search:
        # Fit model
        clf = MoverClassifier(lr_params, l_feature_names)\
            .fit(X_train, y_train)

        # Evaluate
        d_metric = clf.evaluate(X_train, X_test, y_train, y_test)
        
        # Get evaluation info
        param_name = ','.join(['='.join(list(map(str, [k, v]))) for k, v in lr_params.items()])
        d_accuracies = {k: v for k, v in d_metric.items() if k in ['train_accuracy', 'test_accuracy']}
        d_info = {param_name: d_accuracies}
        
        return clf, d_info
    
    best_params, best_score, d_info = None, 0, {}
    for l_params in itertools.product(*[[(k, v) for v in l_v] for k, l_v in lr_grid_params.items()]):
        
        # Fit and evaluate model 
        xgb_grid_instance_params = {**lr_params, **{k: v for k, v in l_params}}
        clf = MoverClassifier(xgb_grid_instance_params, l_feature_names)\
            .fit(X_train, y_train)
        d_metric = clf.evaluate(X_train, X_test, y_train, y_test)

        # Display and track stats
        param_name = ','.join(['='.join(list(map(str, [k, v]))) for k, v in l_params])
        d_accuracies = {k: v for k, v in d_metric.items() if k in ['train_accuracy', 'test_accuracy']}
        print(l_params)
        print(d_accuracies)
        print('______')
        d_info[param_name] = d_accuracies 
        
        # Keep best params
        if best_params is None:
            best_params = xgb_grid_instance_params
            best_score = d_metric['test_accuracy']

        elif best_score < d_metric['test_accuracy']:
            best_params = xgb_grid_instance_params
            best_score = d_metric['test_accuracy']
    
    clf = MoverClassifier(best_params, l_feature_names)\
        .fit(X_train, y_train)
    
    return clf, d_info 

if __name__ == '__main__':
    # Load params and dataset
    d_params = safe_load(pth_parameters.open())['training']
    with open(pth_training_data / 'mover_dataset.pkl', 'rb') as handle:
        dataset = pickle.load(handle)

    # Split train / test
    X_train, y_train, X_test, y_test = dataset.split_train_test(test_ratio=d_params['test_ratio'])

    # Run grid search
    clf, d_info = fit(
        X_train, y_train, X_test, y_test, d_params['lr_params'], 
        d_params['lr_grid_params'], dataset.feature_names, do_grid_search=True
    )

    # Save best model, datasets & logs
    with open(pth_models / 'mover_lr_model.pkl', 'wb') as handle:
        pickle.dump(clf, handle)
        
    with open(pth_models / 'mover_lr_grid_search.json', 'w') as handle:
        json.dump(d_info, handle)

    with open(pth_training_data / 'mover_lr_dataset.pkl' , 'wb') as handle:
        pickle.dump(dataset, handle)
