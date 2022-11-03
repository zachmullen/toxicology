from itertools import product
import streamlit as st
import numpy as np
import pandas as pd
import pingouin as pg
import plotly.express as px

# TODO next version of st.cache will allow passing a custom message to show_spinner, use that instead.
@st.cache(show_spinner=False)
def df_from_file(file) -> pd.DataFrame:
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


def get_color_by_value(column_name: str, dict_df: pd.DataFrame) -> str:
    return dict_df[dict_df['Field Name'] == column_name].iloc[0].Category


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

        # 3. Loop over all output columns with a progress bar, running prcc on each output
        output_cols = dict_df.loc[dict_df['Type'] == 'Output']['Field Name']

    progress = st.progress(0)
    total_cells = len(varying_input_cols) * len(output_cols)
    heatmap = np.zeros(total_cells, dtype=np.float64)
    for i, (in_col, out_col) in enumerate(product(varying_input_cols, output_cols)):
        progress.progress(i / total_cells)
        prcc = pg.partial_corr(data_df, in_col, out_col, method='spearman')
        heatmap[i] = abs(prcc.r)

    heatmap = heatmap.reshape((len(varying_input_cols), len(output_cols)))
    progress.empty()
    fig = px.imshow(
        heatmap,
        labels=dict(x='Output', y='Input', color='|PRCC|'),
        x=output_cols,
        y=varying_input_cols,
        height=45 * len(varying_input_cols),
    )
    st.plotly_chart(fig, use_container_width=True)
