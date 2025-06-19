import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def read_csv(input_path):
    """This function is read the input data path

    Args:
        input_path (csv): CSV file that contains the dataset
    Returns:
       df: Dataframe
    """
    df = pd.read_csv(input_path)
    print(df.head())
    return df

def inspect_data(df):
    print('Head of data:', df.head())
    print('Columns of data:', df.columns)
    print('Describe the data:', df.describe())
    print('Info:', df.info())
    print('Check null data:', df.isna().sum())
    
def prepare_columns(df):   
    df = df.drop(columns = "Unnamed: 32", axis = 1) 
    # Map the string values to binary values
    df["diagnosis"] = df["diagnosis"].map({'M':1, 'B':0})
    df = df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
    return df

def distribution_graph(df):
    # Map numeric diagnosis back to string labels for plotting
    diagnosis_labels = df['diagnosis'].map({0: 'Benign', 1: 'Malignant'})

    fig = px.histogram(x=diagnosis_labels, 
                    color=diagnosis_labels,
                    color_discrete_map={
                        'B': 'rgb(50, 80, 168)',     
                        'M': '#ee8ef5'}
                    )
    fig.update_xaxes(categoryorder="array", categoryarray=['Benign', 'Malignant'])
    fig.update_layout(
        title="Distribution of Diagnosis",
        xaxis_title="Diagnosis",
        yaxis_title="Count of Diagnosis Distribution"
    )
    fig.update_layout(
        width=600,
        height=400,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
    )

    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(
            showline=True,            # show axis line
            linecolor='darkgray',        # axis line color
            showgrid=True,            # show grid lines
            gridcolor='lightgray',    # grid line color
            gridwidth=1
        ),
        yaxis=dict(
            showline=True,
            linecolor='darkgray',
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        )
    )

    fig.show()
    return fig

def correlation_analysis(df):
    corr_matrix = df.corr(numeric_only=True)
    print(df.corr()['diagnosis'] > 0)
    df_numeric = df.copy()
    target_corr = df_numeric.corr()['diagnosis'].drop('diagnosis').sort_values(ascending=False)

    print("Features most correlated with diagnosis:")
    print(target_corr.head(10))

    print("Features negatively correlated with diagnosis:")
    print(target_corr.tail(10))
    
    # Find features with correlation > 0.9 (or <-0.9)
    threshold = 0.9
    corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)

    # Filter out self-correlations
    high_corr_pairs = [(a, b, corr_matrix.loc[a, b]) for a, b in corr_pairs.index if a != b and corr_matrix.loc[a, b] > threshold]

    # Show results
    for a, b, corr_val in high_corr_pairs:
        print(f"{a} & {b} => correlation: {corr_val:.3f}")

def scatter_matrix(df):
    # Map numeric diagnosis back to string labels for plotting
    diagnosis_labels = df['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
    fig = px.scatter_matrix(
        df,
        dimensions=['radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'],
        color=diagnosis_labels, 
        color_discrete_map={
            'Benign': 'rgb(50, 80, 168)',     
            'Malignant': '#ee8ef5'     
    })

    fig.update_layout(
        title=dict(text='Breast Cancer Diagnosis'),
        width=1500,
        height=1500,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        ),
        plot_bgcolor='white'
    )

    fig.show()
    return fig

def distribution_graph(df):
    columns_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean']

    for column in columns_to_plot:
        fig = px.histogram(df, x=df[column], color="diagnosis", marginal="box",
                        color_discrete_map={
                                    0: 'rgb(50, 80, 168)',     
                                    1: '#ee8ef5'})
        fig.update_layout(
        title=dict(text=f"Box plot of {column} by Diagnosis"),
        width=800,
        height=400
        )
        fig.update_layout(
        width=600,
        height=400,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
        )

        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=True,            # show axis line
                linecolor='darkgray',        # axis line color
                showgrid=True,            # show grid lines
                gridcolor='lightgray',    # grid line color
                gridwidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='darkgray',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            )
        )
        fig.show()
        
        return fig
    
def box_plot(df):
    columns_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean']

    for column in columns_to_plot:
        fig = px.box(df, x="diagnosis", 
                    y=column, 
                    color="diagnosis", 
                    color_discrete_map={
                                    0: 'rgb(50, 80, 168)',     
                                    1: '#ee8ef5'}
        )
        fig.update_layout(
            title=dict(text=f"Box plot of {column} by Diagnosis"),
            width=800,
            height=400
        )
        fig.update_layout(
        width=600,
        height=400,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
        )

        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=True,            # show axis line
                linecolor='darkgray',        # axis line color
                showgrid=True,            # show grid lines
                gridcolor='lightgray',    # grid line color
                gridwidth=1
            ),
            yaxis=dict(
                showline=True,
                linecolor='darkgray',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=1
            )
        )
        fig.show()
        return fig
    
def correlation_matrix(df):
    corr_matrix = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Purp',  
        text=corr_matrix.values,  # Add the correlation values as text
        texttemplate="%{z:.2f}",  # Format the text (2 decimal places)
        showscale=True
    ))
    fig.update_layout(title='Correlation Heatmap')
    fig.update_layout(
        width=1000,
        height=800,
        font=dict(
            family="Arial, Courier New, monospace",  # font family
            size=12,                                 # font size (pixels)
            color="darkblue"                         # font color
        )
    )
    fig.show()

    return fig