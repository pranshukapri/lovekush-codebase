import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import os
import re
import numpy as np
import pandas as pd

import pickle
import textwrap  # for wrapping text for plotly charts

import tensorflow_hub as hub
import umap
import hdbscan
from sklearn.metrics import silhouette_score

import plotly.express as px

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(external_stylesheets=external_stylesheets)

dir_name = "/project/data_preprocessed_csv/"
df_metadata = pd.read_csv("/project/preprocessing/metadata.csv", index_col=0)
counts = df_metadata.groupby("case").case.count()
cases = counts[counts > 1].index.values
df_lawyers = pd.read_csv("/project/eda/lawyers_stats.csv", index_col=0)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

app.layout = html.Div(
    [
        html.H1("Deposition Analyser"),
        html.H3("Question Visualiser"),
        html.P(
            "After you select a case and click 'Create visualisation' (and after a bit of time, should be under 20 seconds)"
            " an interactive visualisation should appear. Questions which have a similar mean"
            "ing will appear closer together. You can hover over the points to reveal the questions. You can"
            " zoom in by dragging a rectangle over the part of the graph you want to zoom in on."
            " You can reset the plot by clicking the icon in the shape of a house in the top-right."
        ),
        html.Div(
            dcc.Dropdown(
                id="case_selection",
                options=[{"label": case, "value": case} for case in cases],
                value=cases[0],
                style={"width": "80%"},
            )
        ),
        html.Button("Create visualisation", id="visualisation_button"),
        dcc.Graph(id="question-cluster-graph"),
        html.H3("Similar question finder"),
        html.P(
            "After selecting a case, you can enter"
            " a question in the box below. You will then be showed the top 10 most similar questions "
            " along with their answers, and a 'similarity score' that shows how much the algorithms think"
            " the given questions are similar to the question you entered."
        ),
        dcc.Textarea(
            id="users_question",
            placeholder="Enter a question...",
            style={"width": "100%"},
        ),
        html.Button("Find similar questions", id="similar_questions_button"),
        html.P(id="similar_questions", style={'marginBottom': '5em'}),
        html.H3("Lawyer Stats"),
        html.P(
            "This tool produces statistics that allow you to compare how different lawyers ask"
            " questions."
        ),
        html.Div(
            dcc.Dropdown(
                id="statistic_selection",
                options=[
                    {"label": "Average words per question", "value": "av_num_words"},
                    {"label": "Objection ratio", "value": "objection_ratio"},
                    {"label": "Strike ratio", "value": "strike_ratio"},
                ],
                value="Please select a statistic",
                style={"width": "80%"},
            )
        ),
        html.P(id="lawyer_stats",  style={'marginBottom': '5em'}),
        html.Div(children="", id="intermediate-data", style={"display": "none"}),
    ],
    className="app-site",
)


@app.callback(
    [
        Output("question-cluster-graph", "figure"),
        Output("intermediate-data", "children"),
    ],
    Input("visualisation_button", "n_clicks"),
    State("case_selection", "value"),
)
def process_case_and_create_plot(n, case_selection):
    if not n:
        return empty_plot()

    print(case_selection)

    filenames = df_metadata.loc[df_metadata.case == case_selection, "filename"].values

    # corpus = []
    # for filename in filenames:
    #     try:
    #         df = pd.read_csv(dir_name + filename[:-3] + "csv", index_col=0)
    #         df = df[df.text_type.isin(["q"])]
    #         corpus += df.text.values.tolist()
    #     except:
    #         print(filename)

    # loop through files to create frame containing all questions and answers
    df_aq = pd.DataFrame(columns=["text", "text_type", "filename"])
    for filename in filenames:
        df_individual_aq = pd.DataFrame(columns=["text", "text_type", "filename"])

        df_individual_full = pd.read_csv(dir_name + filename[:-3] + "csv", index_col=0)
        indices = df_individual_full.text_type.isin(["a", "q"])

        df_individual_aq["text"] = df_individual_full.loc[indices, "text"]
        df_individual_aq["text_type"] = df_individual_full.loc[indices, "text_type"]
        df_individual_aq["filename"] = filename[:-4]

        df_aq = pd.concat([df_aq, df_individual_aq], axis=0, ignore_index=True)

    # for each question, determine if it was answered
    # if yes, add that answer to new column
    indices = (df_aq.text_type == "q") & (df_aq.text_type.shift(-1) == "a")
    indices_a = indices.shift(1)
    indices_a[0] = False
    df_aq.loc[indices, "answer"] = df_aq.loc[indices_a, "text"].values
    df_q = df_aq.loc[df_aq.text_type == "q", ["text", "answer", "filename"]]

    # vectorise corpus using Univeral Sentence Embedding
    df_q["vectors"] = embed(df_q.text).numpy().tolist()

    vectors_dim_reduced = umap.UMAP(random_state=0).fit_transform(df_q.vectors.tolist())
    df_q["x"] = vectors_dim_reduced[:, 0]
    df_q["y"] = vectors_dim_reduced[:, 1]

    df_q["clusters"] = hdbscan.HDBSCAN().fit(vectors_dim_reduced).labels_

    # indices = clusters > -1

    df_plot = pd.DataFrame(vectors_dim_reduced, columns=["x", "y"])
    df_plot["Text"] = df_q.text.values
    df_plot["cluster"] = df_q.clusters.values

    # need to manually wrap text for plotly, using html newline tags
    df_plot.Text = df_plot.Text.apply(
        lambda txt: "<br>".join(textwrap.wrap(str(txt), width=30))
    )

    df_plot = df_plot[df_plot.cluster > -1]

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        hover_data=dict(x=False, y=False, Text=True, cluster=False),
        width=600,
        height=600,
        color="cluster",
        color_continuous_scale="rainbow",
    )

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title=f"Visualising the questions from the case {case_selection}",
    )

    fig.update_xaxes(
        linecolor="black",
        mirror=True,
        title=None,
        showticklabels=False,
        linewidth=2,
    )
    fig.update_yaxes(
        linecolor="black",
        mirror=True,
        title=None,
        showticklabels=False,
        linewidth=2,
    )

    print(case_selection)
    return fig, df_q.to_json(date_format="iso", orient="split")


@app.callback(
    Output("similar_questions", "children"),
    Input("similar_questions_button", "n_clicks"),
    [State("intermediate-data", "children"), State("users_question", "value")],
)
def find_similar_questions(n, data, question):
    if not n:
        return ""
    if not data:
        return "Please select a case first"

    df = pd.read_json(data, orient="split")

    q_vector = embed([question])[0].numpy()
    vectors = np.array(df.vectors.tolist())
    df["similarity"] = vectors @ q_vector.T
    df.sort_values(by="similarity", axis=0, ascending=False, inplace=True)

    similar_questions_list = []
    for i in range(10):
        index = df.index[i]
        similarity = df.loc[index, "similarity"]
        question = df.loc[index, "text"]
        answer = df.loc[index, "answer"]
        filename = df.loc[index, "filename"]

        similar_questions_list.append(f"Question similarity score: {similarity}")
        similar_questions_list.append(html.Br())
        similar_questions_list.append(f"Question: {question}")
        similar_questions_list.append(html.Br())
        similar_questions_list.append(f"Answer: {answer}")
        similar_questions_list.append(html.Br())
        similar_questions_list.append(f"Deposition: {filename}")
        similar_questions_list.append(html.Br())
        similar_questions_list.append(html.Br())

    return similar_questions_list


@app.callback(
    Output("lawyer_stats", "children"),
    Input("statistic_selection", "value"),
)
def produce_lawyer_stats(statistic):
    if statistic == 'Please select a statistic':
        return statistic

    top10 = df_lawyers.loc[
        df_lawyers.num_questions > 20, [statistic]
    ].sort_values(by=statistic, axis=0, ascending=False).head(10)

    bottom10 = df_lawyers.loc[
        df_lawyers.num_questions > 20, [statistic]
    ].sort_values(by=statistic, axis=0, ascending=True).head(10)

    lawyer_stats = []

    lawyer_stats.append(html.H6('Highest 10'))
    for i in range(10):
        lawyer_stats.append(f'{round(top10[statistic][i],2)} -- {top10.index[i]}')
        lawyer_stats.append(html.Br())
    lawyer_stats.append(html.Br())
    
    lawyer_stats.append(html.H6('Lowest 10'))
    for i in range(10):
        lawyer_stats.append(f'{round(bottom10[statistic][i],2)} -- {bottom10.index[i]}')
        lawyer_stats.append(html.Br())
    lawyer_stats.append(html.Br())

    return lawyer_stats



def empty_plot():
    fig = px.scatter(
        pd.DataFrame(columns=["x", "y"]),
        x="x",
        y="y",
        width=600,
        height=600,
    )

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        title=f"Empty plot. Please select a case",
    )

    fig.update_xaxes(
        linecolor="black",
        mirror=True,
        title=None,
        showticklabels=False,
        linewidth=2,
    )
    fig.update_yaxes(
        linecolor="black",
        mirror=True,
        title=None,
        showticklabels=False,
        linewidth=2,
    )

    return fig, None


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)