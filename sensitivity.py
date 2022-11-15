from itertools import product
import math
import streamlit as st
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px
from streamlit_plotly_events import plotly_events
from typing import Iterable

# TODO next version of st.cache will allow passing a custom message to show_spinner, use that instead.
@st.cache(show_spinner=False, allow_output_mutation=True)
def df_from_file(file) -> pd.DataFrame:
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


@st.cache(show_spinner=False)
def compute_prcc_cell(df: pd.DataFrame, in_col: str, out_col: str) -> float:
    return float(pg.partial_corr(df, in_col, out_col, method='spearman').r)


@st.cache(show_spinner=False, suppress_st_warning=True)
def compute_prcc_heatmap(df: pd.DataFrame, input_cols: Iterable[str], output_cols: Iterable[str]) -> np.ndarray:
    progress_message = st.empty()
    progress = st.progress(0)
    total_cells = len(input_cols) * len(output_cols)

    heatmap = np.zeros(total_cells, dtype=np.float64)
    for i, (in_col, out_col) in enumerate(product(input_cols, output_cols)):
        progress_message.text(f'Computing PRCC: {i + 1} of {total_cells}')
        progress.progress(i / total_cells)
        prcc = compute_prcc_cell(df, in_col, out_col)
        heatmap[i] = abs(prcc)

    progress_message.empty()
    progress.empty()

    return heatmap.reshape((len(input_cols), len(output_cols)))


def get_color_by_value(column_name: str, dict_df: pd.DataFrame) -> str:
    category = dict_df[dict_df['Field Name'] == column_name].iloc[0].Category
    if isinstance(category, float) and math.isnan(category):
        return 'Unknown'
    return category


st.set_page_config(layout="wide")
st.title('Toxicology sensitivity analysis')

upload_container = st.empty()
with upload_container.container():
    dict_file = st.file_uploader('Select data dictionary', type=['csv', 'xlsx'])
    data_file = st.file_uploader('Select data', type=['csv', 'xlsx'])

if data_file and dict_file:
    upload_container.empty()

    with st.spinner('Ingesting data...'):
        data_df = df_from_file(data_file)
        dict_df = df_from_file(dict_file)

        # 1. Find all input columns
        input_cols = dict_df.loc[dict_df['Type'] == 'Input']['Field Name']

        # 2. Ignore input columns that have the same value in every row
        varying_input_cols = [col for col in input_cols if data_df[col].nunique() > 1]

        # 3. Find all output columns
        output_cols = dict_df.loc[dict_df['Type'] == 'Output']['Field Name']

    with st.expander('Data dictionary'):
        dict_df

    with st.expander('Sample data rows'):
        data_df[:10]

    # Compute PRCC for all input / output combinations
    heatmap = compute_prcc_heatmap(data_df, varying_input_cols, output_cols)
    fig = px.imshow(
        heatmap,
        labels=dict(x='Output', y='Input', color='|PRCC|'),
        x=output_cols,
        y=varying_input_cols,
        color_continuous_scale=[(0, 'white'), (1, 'red')]
    )
    selected_points = plotly_events(fig, override_height=45 * len(varying_input_cols))

    if selected_points:
        selected_point = selected_points[0]
        selected_input, selected_output = selected_point['y'], selected_point['x']

        # show two bar charts: all inputs against the selected output, and all outputs against the selected input
        results = {
            output_col: compute_prcc_cell(data_df, selected_input, output_col)
            for output_col in output_cols
        }
        result_list = [
            [k, v, get_color_by_value(k, dict_df)]
            for k, v in results.items()
        ]
        plot_df = pd.DataFrame(result_list, columns=['Parameter', 'PRCC', 'Category'])
        fig = px.bar(plot_df, x='Parameter', y='PRCC', title=f'Input sensitivity: {selected_input}', color='Category')
        fig.update_layout(xaxis_categoryorder='total descending')
        st.plotly_chart(fig, use_container_width=True)

        results = {
            input_col: compute_prcc_cell(data_df, input_col, selected_output)
            for input_col in varying_input_cols
        }
        result_list = [
            [k, v, get_color_by_value(k, dict_df)]
            for k, v in results.items()
        ]
        plot_df = pd.DataFrame(result_list, columns=['Parameter', 'PRCC', 'Category'])
        fig = px.bar(plot_df, x='Parameter', y='PRCC', title=f'Output sensitivity: {selected_output}', color='Category')
        fig.update_layout(xaxis_categoryorder='total descending')
        st.plotly_chart(fig, use_container_width=True)
