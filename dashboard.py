import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_diabetes
import plotly.express as px
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load built-in datasets
iris = load_iris()
wine = load_wine()
diabetes = load_diabetes()

# Create DataFrames with enhanced features
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]
iris_df['species_numeric'] = iris_df['species'].apply(lambda x: {'setosa': 0, 'versicolor': 1, 'virginica': 2}[x])

# Add PCA and t-SNE features for iris
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris.data)

# PCA for iris
iris_pca_obj = PCA(n_components=2)
iris_pca = iris_pca_obj.fit_transform(iris_scaled)
iris_df['PCA1'] = iris_pca[:, 0]
iris_df['PCA2'] = iris_pca[:, 1]

# t-SNE for iris
tsne = TSNE(n_components=2, random_state=42)
iris_tsne = tsne.fit_transform(iris_scaled)
iris_df['tSNE1'] = iris_tsne[:, 0]
iris_df['tSNE2'] = iris_tsne[:, 1]

# Wine dataset
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['wine_class'] = wine.target_names[wine.target]
numeric_wine_df = wine_df.select_dtypes(include=[np.number])

# PCA for wine
wine_scaled = scaler.fit_transform(wine.data)
wine_pca_obj = PCA(n_components=2)
wine_pca = wine_pca_obj.fit_transform(wine_scaled)
wine_df['PCA1'] = wine_pca[:, 0]
wine_df['PCA2'] = wine_pca[:, 1]

# Diabetes dataset
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target
diabetes_df['target_category'] = pd.cut(diabetes_df['target'], bins=5, labels=['Low', 'Moderate Low', 'Medium', 'Moderate High', 'High'])

# Initialize Dash app with better layout
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Define app layout with improved structure
app.layout = dbc.Container([
    html.Div([
        html.H1("Advanced Data Visualization Dashboard", className="mb-4 text-center"),
        html.P("Explore and analyze built-in datasets with interactive visualizations", 
               className="text-center text-muted mb-4"),
    ], className="header-section"),
    
    dbc.Tabs([
        # Tab 1 - Iris Dataset
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("Interactive Scatter Plot", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='iris-x-axis',
                                        options=[{'label': col, 'value': col} for col in iris.feature_names + ['PCA1', 'tSNE1']],
                                        value='sepal length (cm)',
                                        clearable=False
                                    ),
                                ], md=6),
                                dbc.Col([
                                    dcc.Dropdown(
                                        id='iris-y-axis',
                                        options=[{'label': col, 'value': col} for col in iris.feature_names + ['PCA2', 'tSNE2']],
                                        value='sepal width (cm)',
                                        clearable=False
                                    ),
                                ], md=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.RadioItems(
                                        id='iris-color',
                                        options=[
                                            {'label': 'By Species', 'value': 'species_numeric'},
                                            {'label': 'By Sepal Length', 'value': 'sepal length (cm)'},
                                            {'label': 'By Petal Width', 'value': 'petal width (cm)'}
                                        ],
                                        value='species_numeric',
                                        inline=True,
                                        className="mb-2"
                                    ),
                                ])
                            ]),
                            dcc.Graph(id='iris-scatter', style={'height': '400px'})
                        ])
                    ], className="mb-4"),
                    
                    html.H4("Statistical Summary", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='iris-stats-table',
                                columns=[{"name": i, "id": i} for i in ['stat'] + iris.feature_names],
                                style_table={'overflowX': 'auto'},
                                style_cell={'textAlign': 'left', 'padding': '5px'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                            )
                        ])
                    ])
                ], md=6),
                
                dbc.Col([
                    html.H4("Distribution Analysis", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='iris-feature',
                                options=[{'label': col, 'value': col} for col in iris.feature_names],
                                value='sepal length (cm)',
                                clearable=False,
                                className="mb-3"
                            ),
                            dcc.Graph(id='iris-boxplot', style={'height': '300px'})
                        ])
                    ], className="mb-4"),
                    
                    html.H4("Feature Correlation", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.imshow(
                                    iris_df[iris.feature_names].corr(),
                                    title='Feature Correlation Heatmap',
                                    color_continuous_scale='RdBu',
                                    zmin=-1,
                                    zmax=1
                                ),
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], md=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Multivariate Analysis", className="mb-3"),
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(
                                figure=px.parallel_coordinates(
                                    iris_df,
                                    color='species_numeric',
                                    dimensions=iris.feature_names,
                                    color_continuous_scale=px.colors.diverging.Tealrose,
                                    title='Parallel Coordinates Plot'
                                )
                            )
                        ], label="Parallel Coordinates"),
                        dbc.Tab([
                            dcc.Graph(
                                figure=px.scatter_matrix(
                                    iris_df,
                                    dimensions=iris.feature_names,
                                    color='species_numeric',
                                    title='Scatter Matrix'
                                )
                            )
                        ], label="Scatter Matrix"),
                        dbc.Tab([
                            dcc.Graph(
                                figure=px.scatter(
                                    iris_df,
                                    x='PCA1',
                                    y='PCA2',
                                    color='species_numeric',
                                    title='PCA Visualization'
                                )
                            )
                        ], label="PCA"),
                        dbc.Tab([
                            dcc.Graph(
                                figure=px.scatter(
                                    iris_df,
                                    x='tSNE1',
                                    y='tSNE2',
                                    color='species_numeric',
                                    title='t-SNE Visualization'
                                )
                            )
                        ], label="t-SNE")
                    ])
                ])
            ])
        ], label="Iris Dataset", tabClassName="ml-auto"),
        
        # Tab 2 - Wine Dataset
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("Class Distribution", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.pie(
                                    wine_df,
                                    names='wine_class',
                                    title='Wine Class Distribution',
                                    hole=0.4,
                                    color_discrete_sequence=px.colors.qualitative.Pastel
                                ),
                                style={'height': '400px'}
                            )
                        ])
                    ], className="mb-4"),
                    
                    html.H4("Feature Importance", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(
                                    pd.DataFrame({
                                        'features': wine.feature_names,
                                        'importance': np.abs(wine_pca_obj.components_[0])
                                    }).sort_values('importance', ascending=False),
                                    x='features',
                                    y='importance',
                                    title='PCA Feature Importance (First Component)',
                                    color='importance',
                                    color_continuous_scale='Blues'
                                ),
                                style={'height': '400px'}
                            )
                        ])
                    ])
                ], md=4),
                
                dbc.Col([
                    html.H4("Feature Analysis", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='wine-feature-selector',
                                options=[{'label': col, 'value': col} for col in wine.feature_names[:6]],
                                value=wine.feature_names[0],
                                clearable=False,
                                className="mb-3"
                            ),
                            dcc.Graph(id='wine-feature-plot', style={'height': '350px'})
                        ])
                    ], className="mb-4"),
                    
                    html.H4("PCA Visualization", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.scatter(
                                    wine_df,
                                    x='PCA1',
                                    y='PCA2',
                                    color='wine_class',
                                    title='PCA of Wine Dataset',
                                    symbol='wine_class',
                                    size_max=15
                                ),
                                style={'height': '350px'}
                            )
                        ])
                    ])
                ], md=8)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Correlation Analysis", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.imshow(
                                    numeric_wine_df.corr(),
                                    title='Wine Features Correlation Heatmap',
                                    color_continuous_scale='RdBu',
                                    zmin=-1,
                                    zmax=1
                                ),
                                style={'height': '600px'}
                            )
                        ])
                    ])
                ])
            ])
        ], label="Wine Dataset"),
        
        # Tab 3 - Diabetes Dataset
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("Target Distribution", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.histogram(
                                    diabetes_df,
                                    x='target',
                                    nbins=30,
                                    title='Diabetes Progression Distribution',
                                    marginal='box',
                                    color_discrete_sequence=['#636EFA']
                                ),
                                style={'height': '400px'}
                            )
                        ])
                    ], className="mb-4"),
                    
                    html.H4("Feature Relationships", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='diabetes-feature',
                                options=[{'label': col, 'value': col} for col in diabetes.feature_names],
                                value='bmi',
                                clearable=False,
                                className="mb-3"
                            ),
                            dcc.RadioItems(
                                id='diabetes-plot-type',
                                options=[
                                    {'label': 'Scatter Plot', 'value': 'scatter'},
                                    {'label': 'Violin Plot', 'value': 'violin'},
                                    {'label': 'Box Plot', 'value': 'box'}
                                ],
                                value='scatter',
                                inline=True,
                                className="mb-3"
                            ),
                            dcc.Graph(id='diabetes-plot', style={'height': '350px'})
                        ])
                    ])
                ], md=6),
                
                dbc.Col([
                    html.H4("Multivariate Analysis", className="mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.scatter_matrix(
                                    diabetes_df,
                                    dimensions=diabetes.feature_names[:4] + ['target'],
                                    color='target_category',
                                    title='Scatter Matrix of Key Features',
                                    opacity=0.7
                                ),
                                style={'height': '800px'}
                            )
                        ])
                    ])
                ], md=6)
            ])
        ], label="Diabetes Dataset")
    ], className="mb-4"),
    
    # Footer
    html.Footer([
        html.Div([
            html.P("© 2025 Data Visualization Dashboard", className="text-center text-muted"),
            html.P("Built by HPDS Team", className="text-center text-muted")
        ], className="py-3 border-top")
    ])
], fluid=True)

# Callbacks for interactive plots
@app.callback(
    Output('iris-scatter', 'figure'),
    [Input('iris-x-axis', 'value'),
     Input('iris-y-axis', 'value'),
     Input('iris-color', 'value')]
)
def update_iris_scatter(x_axis, y_axis, color_by):
    fig = px.scatter(
        iris_df,
        x=x_axis,
        y=y_axis,
        color=color_by,
        title=f'{x_axis} vs {y_axis}',
        color_continuous_scale=px.colors.diverging.Tealrose if color_by != 'species_numeric' else None,
        hover_data={'species': True}
    )
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('iris-boxplot', 'figure'),
    [Input('iris-feature', 'value')]
)
def update_iris_boxplot(feature):
    fig = px.box(
        iris_df,
        x='species',
        y=feature,
        title=f'{feature} Distribution by Species',
        color='species',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('iris-stats-table', 'data'),
    [Input('iris-feature', 'value')]
)
def update_iris_stats_table(feature):
    stats = iris_df[iris.feature_names].describe().reset_index()
    stats.columns = ['stat'] + iris.feature_names
    return stats.to_dict('records')

@app.callback(
    Output('wine-feature-plot', 'figure'),
    [Input('wine-feature-selector', 'value')]
)
def update_wine_feature_plot(feature):
    fig = px.box(
        wine_df,
        x='wine_class',
        y=feature,
        title=f'{feature} Distribution by Wine Class',
        color='wine_class',
        points='all'
    )
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('diabetes-plot', 'figure'),
    [Input('diabetes-feature', 'value'),
     Input('diabetes-plot-type', 'value')]
)
def update_diabetes_plot(feature, plot_type):
    if plot_type == 'scatter':
        fig = px.scatter(
            diabetes_df,
            x=feature,
            y='target',
            trendline='ols',
            title=f'{feature} vs Diabetes Progression',
            color='target_category',
            opacity=0.7
        )
    elif plot_type == 'violin':
        fig = px.violin(
            diabetes_df,
            x='target_category',
            y=feature,
            title=f'{feature} Distribution by Target Category',
            box=True,
            points='all'
        )
    else:  # box plot
        fig = px.box(
            diabetes_df,
            x='target_category',
            y=feature,
            title=f'{feature} Distribution by Target Category',
            color='target_category'
        )
    
    fig.update_layout(transition_duration=500)
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)