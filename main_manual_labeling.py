# This file runs an experiment using manual (human) labeling.
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, html

from src.classifier import LogisticRegressionClassifier
from src.seals_manual import SEALSManualAlgorithm
from src.selection_strategy import MaxEntropySelectionStrategy

# Initialize SEALS
classifier = LogisticRegressionClassifier()
selection = MaxEntropySelectionStrategy()
baselines = []
seals = SEALSManualAlgorithm(
    classifier, selection, baseline_algorithms=baselines, eval_class="/m/01bdy"
)
first_uri, first_label = seals.step()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.Div(html.H1("Manual Labeling of SEALS Algorithm")),
        html.Div(html.H3("Class: Bowling")),
        html.Div(
            html.Img(
                id="label-image", src=first_uri, style={"height": "600px"}
            )
        ),
        html.Div(
            [
                dbc.Button("YES", id="yes-button", n_clicks=0),
                dbc.Button("NO", id="no-button", n_clicks=0),
                html.H5(
                    children=f"OI Label: {'YES' if first_label==1 else 'NO'}",
                    id="label-text",
                ),
            ]
        ),
        dbc.Progress(id="progress-bar", label="0 / 2000", value=0),
    ]
)

total_clicks = [0, 0]


@app.callback(
    Output("label-image", "src"),
    Output("progress-bar", "label"),
    Output("progress-bar", "value"),
    Output("label-text", "children"),
    Input("yes-button", "n_clicks"),
    Input("no-button", "n_clicks"),
)
def yes_or_no_button(n_clicks_yes, n_clicks_no):
    if n_clicks_yes > total_clicks[0]:
        seals.label = 1
        total_clicks[0] += 1
    elif n_clicks_no > total_clicks[1]:
        seals.label = 0
        total_clicks[1] += 1
    uri, oi_label = seals.step()
    progress = seals.labeled_set.size / 20
    label = f"OI Label: {'YES' if oi_label==1 else 'NO'}"
    return uri, f"{seals.labeled_set.size} / 2000", progress, label


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
