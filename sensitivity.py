import math
import streamlit as st
import pandas as pd
import pingouin as pg
import plotly.express as px


@st.cache
def df_from_file(file) -> pd.DataFrame:
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


def get_color_by_value(column_name: str, dict_df: pd.DataFrame) -> str:
    return dict_df[dict_df['Field Name'] == column_name].iloc[0].Category


def run_prcc(dict_df: pd.DataFrame, data_df: pd.DataFrame, output_name: str) -> dict[str, float]:
    input_names = dict_df.loc[dict_df['Type'] == 'Input']['Field Name'].to_list()
    return {
        input_name: float(pg.partial_corr(data_df, input_name, output_name, method='spearman').r)
        for input_name in input_names
    }


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

        output_cols = dict_df.loc[dict_df['Type'] == 'Output']['Field Name'].to_list()

    output_name = st.selectbox('Select output parameter:', options=[''] + output_cols)

    if output_name:
        with st.spinner(f'Running PRCC on {output_name}'):
            results = run_prcc(dict_df, data_df, output_name)
            result_list = [
                [k, v, get_color_by_value(k, dict_df)]
                for k, v in results.items() if not math.isnan(v)
            ]
            plot_df = pd.DataFrame(result_list, columns=['Parameter', 'PRCC', 'Category'])
            fig = px.bar(plot_df, x='Parameter', y='PRCC', title=f'PRCC results: {output_name}', color='Category')
            fig.update_layout(xaxis_categoryorder='total descending')
            fig
