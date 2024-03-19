from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, Normalizer

from optuna import Trial
from category_encoders import WOEEncoder

def instantiate_numerical_simple_imputer(trial : Trial, fill_value : int=-1) -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'numerical_strategy', ['mean', 'median', 'most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_categorical_simple_imputer(trial : Trial, fill_value : str='missing') -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'categorical_strategy', ['most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_KNNImputer(trial: Trial) -> KNNImputer:
  params = { 
    'weights': trial.suggest_categorical('weights',['uniform', 'distance']),
    'n_neighbors': trial.suggest_int('n_neighbors', 1, 10),
  }
  return KNNImputer(**params)

def instantiate_woe_encoder(trial : Trial) -> WOEEncoder:
  params = {
    'sigma': trial.suggest_float('sigma', 0.001, 5),
    'regularization': trial.suggest_float('regularization', 0, 5),
    'randomized': trial.suggest_categorical('randomized', [True, False])
  }
  return WOEEncoder(**params)

def instantiate_robust_scaler(trial : Trial) -> RobustScaler:
  params = {
    'with_centering': trial.suggest_categorical(
      'with_centering', [True, False]
    ),
    'with_scaling': trial.suggest_categorical(
      'with_scaling', [True, False]
    )
  }
  return RobustScaler(**params)

def instatniate_normalizer(trial: Trial) -> Normalizer:
  norm = trial.suggest_categorical('norm', ['l1', 'l2', 'max'])
  return Normalizer(norm=norm)