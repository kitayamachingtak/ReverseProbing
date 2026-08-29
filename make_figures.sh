#!/bin/bash
# Regenerate every figure from outputs/. Run after train_classifiers.py.
set -e

python draw_image/fig_auprc_bars.py
python draw_image/fig_feature_curve.py
python draw_image/fig_importance_heatmap.py --classifier XGBoost
python draw_image/fig_importance_heatmap.py --classifier CatBoost
python draw_image/fig_importance_heatmap.py --classifier LogisticRegression \
    --groups signal --config max

echo "figures written to figures/"
