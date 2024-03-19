from sklearn.linear_model import LogisticRegression
from optuna import Trial

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as XGBClassifier

def instantiate_xgb(trial : Trial) -> XGBClassifier:
    params={
            'eval_metric': 'auc',
            'lambda': trial.suggest_float( 'xgb_lambda',1e-06, 1e-04, log=True),
            'alpha':  trial.suggest_float( 'xgb_alpha',0.0001, 0.01, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 1, 20),
            'eta': trial.suggest_float( 'xgb_eta',0.0001, 0.01, log=True),
            'gamma': trial.suggest_float( 'xgb_gamma',1e-08, 1e-06, log=True),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 1000, log=True),
            'learning_rate': trial.suggest_float( 'xgb_learning_rate',0.001, 1, log=True),
            'tree_method': 'hist', 
            'device': 'cuda',
           }
    return XGBClassifier(**params)

def instantiate_catb(trial : Trial) -> CatBoostClassifier:
    params = {
        'logging_level': 'Silent', 
        'random_seed': 0, 
        'iterations':  trial.suggest_int('catb_iterations', 50, 1000, log=True),
        'depth': trial.suggest_int('catb_depth', 10, 200),
        'min_data_in_leaf': trial.suggest_int('catb_min_data_in_leaf', 10, 200),
        'learning_rate': trial.suggest_float( 'catb_learning_rate',0.001, 1, log=True),
        'subsample': trial.suggest_float( 'catb_subsample',0.01, 1, log=True),
        'random_strength': trial.suggest_float( 'catb_random_strength',0.01, 1, log=True),
        'eval_metric' : 'AUC',
        'grow_policy': 'Lossguide',
        'bootstrap_type' : 'Bernoulli',
        'task_type':"GPU"
}
    return CatBoostClassifier(**params)

def instantiate_lgbm(trial : Trial) -> LGBMClassifier:
    params = {
        'metric': 'auc', 
        'max_depth': trial.suggest_int('lgbm_max_depth', 1, 20),
        'min_child_samples': trial.suggest_int('lgbm_max_depth', 1, 20), 
        'learning_rate': trial.suggest_float( 'lgbm_learning_rate',0.001, 1, log=True),
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 1000, log=True),
        'min_child_weight': trial.suggest_int('lgbm_max_depth', 1, 20), 
        'subsample': trial.suggest_float( 'lgbm_subsample',0.01, 1, log=True),
        'colsample_bytree':trial.suggest_float( 'lgbm_colsample_bytree',0.01, 1, log=True),
        'reg_alpha':trial.suggest_float( 'lgbm_reg_alpha',0.01, 1, log=True), 
        'reg_lambda': trial.suggest_float( 'lgbm_reg_lambda',0.01, 1, log=True),
        'random_state': 42,
        'verbose': -1,
        'device':"gpu" 
    }
    return LGBMClassifier(**params)

def instantiate_lr(trial : Trial) -> LogisticRegression:
    params = {
        'max_iter': trial.suggest_int('lr_max_iter', 3000, 10000, log=True),
        'C': trial.suggest_float('lr_C', 0.001, 10, log=True),
        'tol': trial.suggest_float('lr_tol', 0.00001, 1, log=True),
        'warm_start': True,
        'solver':'saga',
    }
    return LogisticRegression(**params)