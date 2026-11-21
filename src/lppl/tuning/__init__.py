"""Hyperparameter tuning with Optuna (search), Ray Tune (execution) and MLflow (tracking).

Two studies (docs/tuning.md):

* ``ga``: tune the genetic algorithm on a fixed benchmark of windows (refits every trial).
* ``indicator``: tune qualification filters and the window grid on fits stored by a scan
  (no refitting; the detector is scored against forward drawdowns with a time split).

Heavy dependencies (optuna, ray, mlflow) are imported lazily so the base image and the
core package work without them.
"""
