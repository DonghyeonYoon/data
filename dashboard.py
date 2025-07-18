import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import gradio as gr
from pathlib import Path
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[0]
DATA_PATH = ROOT / 'data' / 'processed' / 'gender_feature_mart.parquet'

df = pd.read_parquet(DATA_PATH)


def _prepare_data():
    def tokens(row):
        t = [f"site_{k}" for k in json.loads(row.site_cnt_map).keys()]
        t += [f"cat1_{c}" for c in row.cat1_arr]
        t += [f"cat2_{c}" for c in row.cat2_arr]
        return t

    hasher = FeatureHasher(n_features=2 ** 20, input_type='string', alternate_sign=False)
    X_sparse = hasher.transform(df.apply(tokens, axis=1))
    dense_cols = [
        'days_since_last',
        'visit_days_30d',
        'num_sites',
        'night_ratio',
        'weekend_ratio',
    ]
    X_dense = sp.csr_matrix(df[dense_cols].to_numpy(dtype=np.float32))
    X = sp.hstack([X_sparse, X_dense]).tocsr()
    y = df['label'].astype(int).to_numpy()
    w = df['label_weight'].to_numpy()
    return train_test_split(X, y, w, test_size=0.2, random_state=42, stratify=y)


X_tr, X_te, y_tr, y_te, w_tr, w_te = _prepare_data()


def train_and_evaluate(num_leaves, learning_rate, feature_fraction, bagging_fraction):
    params = dict(
        objective='binary',
        metric='auc',
        is_unbalance=True,
        num_leaves=int(num_leaves),
        learning_rate=float(learning_rate),
        feature_fraction=float(feature_fraction),
        bagging_fraction=float(bagging_fraction),
        bagging_freq=5,
    )

    dtrain = lgb.Dataset(X_tr, y_tr, weight=w_tr)
    dvalid = lgb.Dataset(X_te, y_te, weight=w_te, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(80)],
        verbose_eval=False,
    )

    proba = model.predict(X_te, num_iteration=model.best_iteration)
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    f1 = f1_score(y_te, pred)

    cm = confusion_matrix(y_te, pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1'],
        ax=ax,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.close(fig_cm)

    importance = model.feature_importance('gain')
    names = [f"f{i}" for i in range(len(importance))]
    fi = pd.Series(importance, index=names).sort_values(ascending=False).head(10)
    fig_fi, ax = plt.subplots(figsize=(6, 4))
    fi.sort_values().plot.barh(ax=ax)
    ax.set_xlabel('Gain')
    plt.close(fig_fi)

    sample_size = min(2000, X_te.shape[0])
    explainer = shap.Explainer(model)
    shap_values = explainer(X_te[:sample_size])
    shap.plots.bar(shap_values, show=False)
    fig_shap = plt.gcf()
    plt.close(fig_shap)

    metrics_md = f"**AUC:** {auc:.3f}  |  **F1:** {f1:.3f}"
    return metrics_md, fig_cm, fig_fi, fig_shap


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown('# LightGBM Dashboard')
        with gr.Row():
            with gr.Column(scale=1):
                num_leaves_slider = gr.Slider(16, 256, value=64, step=16, label='num_leaves')
                lr_slider = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label='learning_rate')
                ff_slider = gr.Slider(0.5, 1.0, value=0.8, step=0.05, label='feature_fraction')
                bf_slider = gr.Slider(0.5, 1.0, value=0.8, step=0.05, label='bagging_fraction')
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem('Metrics'):
                        metrics_text = gr.Markdown()
                        cm_plot = gr.Plot()
                    with gr.TabItem('Feature Importance'):
                        fi_plot = gr.Plot()
                    with gr.TabItem('SHAP'):
                        shap_plot = gr.Plot()
        inputs = [num_leaves_slider, lr_slider, ff_slider, bf_slider]
        outputs = [metrics_text, cm_plot, fi_plot, shap_plot]
        demo.load(train_and_evaluate, inputs=inputs, outputs=outputs)
        num_leaves_slider.change(train_and_evaluate, inputs=inputs, outputs=outputs)
        lr_slider.change(train_and_evaluate, inputs=inputs, outputs=outputs)
        ff_slider.change(train_and_evaluate, inputs=inputs, outputs=outputs)
        bf_slider.change(train_and_evaluate, inputs=inputs, outputs=outputs)
    return demo


if __name__ == '__main__':
    app = build_interface()
    app.launch(server_name='localhost', server_port=5006)
