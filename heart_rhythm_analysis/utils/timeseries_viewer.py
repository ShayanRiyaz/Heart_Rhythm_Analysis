import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def make_window_figure(window, specs):
    """
    window : one row of your DataFrame (a dict-like with arrays + fs keys)
    specs  : list of dicts, each with:
      - key        : column name in window for the signal array
      - fs_key     : column name in window for that signal’s sampling rate
      - subplot    : integer subplot index (you choose)
      - legend     : (optional) legend label for that trace
      - group_title: (optional) title for the whole subplot (overrides auto-title)
    """
    # 1) figure out the distinct subplot numbers, in sorted order
    plot_ids = sorted({s['subplot'] for s in specs})
    id_to_row = {pid: i+1 for i, pid in enumerate(plot_ids)}

    # 2) assemble a title for each subplot
    subplot_titles = []
    for pid in plot_ids:
        # if any spec provides a group_title, take the first one
        gt = next((s['group_title'] for s in specs 
                   if s['subplot']==pid and 'group_title' in s), None)
        if gt:
            title = gt
        else:
            # else join all legend names for that subplot
            legends = [s.get('legend', s['key']) for s in specs if s['subplot']==pid]
            title = " & ".join(legends)
        subplot_titles.append(title)

    # 3) build the subplot figure
    fig = make_subplots(
        rows=len(plot_ids), cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08
    )

    # 4) add each trace into its assigned row
    for s in specs:
        y  = window[s['key']]
        fs = window[s['fs_key']]
        t  = np.arange(len(y)) / fs
        row = id_to_row[s['subplot']]

        fig.add_trace(
            go.Scatter(
                x=t, y=y,
                mode="lines",
                name=s.get('legend', s['key'])
            ),
            row=row, col=1
        )

        fig.update_xaxes(title_text="Time (s)", row=row, col=1)
        fig.update_yaxes(title_text="Amplitude", row=row, col=1)

    fig.update_layout(
        height=300 * len(plot_ids),
        title_text=f"Subject {window['subject']} – Window {window['window_id']}",
        showlegend=True
    )

    return fig

def create_time_series_viewer(
    df: pd.DataFrame,
    figure_fn,
    specs,
    index_label_fn=None,
    dropdown_id="ts-dropdown",
    graph_id="ts-graph"
):
    # fallback if no custom label function provided
    if index_label_fn is None:
        index_label_fn = lambda idx, row: str(idx)

    # build options using index_label_fn
    options = [
        {"label": index_label_fn(idx, row), "value": idx}
        for idx, row in df.iterrows()
    ]

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Time-Series Viewer"),
        dcc.Dropdown(
            id=dropdown_id,
            options=options,
            value=options[0]["value"],
            clearable=False,
            style={"width": "60%", "margin-bottom": "1em"}
        ),
        dcc.Graph(id=graph_id)
    ])

    @app.callback(
        Output(graph_id, "figure"),
        Input(dropdown_id, "value")
    )
    def _update(idx):
        window = df.loc[idx]
        return figure_fn(window)

    return app


    # def update(selected_idx, overlay_keys):
    #     window = df.loc[selected_idx]

    #     # If user has checked one or more keys, build a single overlaid plot
    #     if overlay_keys:
    #         fig = go.Figure()
    #         for key in overlay_keys:
    #             # find the matching fs_key in specs
    #             fs_key = next(s["fs_key"] for s in specs if s["key"] == key)
    #             y  = window[key]
    #             t  = np.arange(len(y)) / window[fs_key]
    #             fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=key))
    #         fig.update_layout(
    #             title=f"Subject {window.subject} – Window {window.window_id} (overlay)",
    #             xaxis_title="Time (s)",
    #             yaxis_title="Amplitude",
    #             height=600,
    #             showlegend=True
    #         )
    #         return fig

    #     # else: fall back to your standard multi-subplot view
    #     return figure_fn(window)

    # return app