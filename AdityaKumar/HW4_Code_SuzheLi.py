#%%

# The dataset that will be used for part of this LAB is CONVENIENT_global_confirmed_cases.csv”. 
# Need to develop a one python file that creates multiple Taps (one tap for each question) in this assignment.

#Create a dashboard with multiple tabs that each tap accommodates each question in this LAB.

# The final python file needs to be deployed through Google cloud (GCP) 
# and a working link must be provided in the report for grading.


#%%

########################################## Import Packages ############################################

import numpy as np
import pandas as pd
import math as math
from scipy.fft import fft
import plotly.express as px
import plotly.graph_objs as go
import dash as dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

########################################## End Here ############################################



########################################## Question 1 Preparation ############################################
# Step 0: 
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url)
## Step 1: Clean the dataset:
df.isna().sum().sum()
print(df.shape)
print(f'Originally, the dataset totally has {df.isna().sum().sum()} nan/missing-values')
print('')
df1 = df.copy()
df1.dropna(inplace=True)
print(f'After dropping, the dataset totally has {df1.isna().sum().sum()} nan/missing-values')
print(df1.shape)
## Step 2: Create 'China_sum' column:
df1['China_sum'] = df1.iloc[0:,57:90].astype(float).sum(axis=1)
# Originally the Dtype is 'object'
print(df1['China_sum'].info())
print('')
print(df1['China_sum'].head())
## Step 3: Create 'United_Kingdom_sum' column: 
df1['United Kingdom_sum'] = df1.iloc[0:, 249:260].astype(float).sum(axis=1)
print(df1['United Kingdom_sum'].info())
print(df1['United Kingdom_sum'].head())
## Step 4: Extract the column we need & Create new Dataframe
df_test = df1[['Country/Region', 'US', 'Brazil', 'United Kingdom_sum', 'China_sum', 'India', 'Italy', 'Germany']]
date = pd.date_range(df_test['Country/Region'][1], df_test['Country/Region'][len(df_test)], periods=len(df_test))
df_test.index = date # df1['Country/Region']
df_test = df_test.drop('Country/Region', axis=1)

########################################## End Here ############################################



########################################## Question 6 Preparation ############################################

import base64
image = 'network.png'
encoded_image = base64.b64encode(open(image, 'rb').read()).decode('ascii')

########################################## End Here ############################################



########################################## Fianl Dash-App setup ############################################

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

hw4_app = dash.Dash('HW4', external_stylesheets=external_stylesheets)

server = hw4_app.server

hw4_app.layout = html.Div([
    html.H1('Homework 4', style={'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id='hw4-questions', children=[
        dcc.Tab(label='Question 1', value='q1'),
        dcc.Tab(label='Question 2', value='q2'),
        dcc.Tab(label='Question 3', value='q3'),
        dcc.Tab(label='Question 4', value='q4'),
        dcc.Tab(label='Question 5', value='q5'),
        dcc.Tab(label='Question 6', value='q6'),
    ]),
    html.Div(id='final-layout'),
])


## Question 1 Tab layout: 
question1_layout = html.Div([
    html.Br(),
    html.H1('Global confirmed COVID19 cases', style={'textAlign': 'center'}),
    
    html.H2('Pick the country Name:'),
    dcc.Dropdown(id='q1-dropdown', options=[
            {'label':'US', 'value':'US'},
            {'label':'Brazil', 'value':'Brazil'},
            {'label':'United Kingdom', 'value':'United Kingdom_sum'} ,
            {'label':'China', 'value':'China_sum'},
            {'label':'India', 'value':'India'},
            {'label':'Italy', 'value':'Italy'},
            {'label':'Germany', 'value':'Germany'},
            ], 
            multi=True, 
            clearable= False,
            value= 'US'), 
    dcc.Graph(id='q1-plot'),
])
### Question 1 Callback: 
@hw4_app.callback(
    Output(component_id='q1-plot', component_property='figure'),
    [Input(component_id='q1-dropdown', component_property='value')]
)
def update_q1(countries):
    fig = px.line( data_frame= df_test, 
                  x= df_test.index, 
                  y= countries,
                  height=500,
                  width=1000
                  )
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Confirmed cases')
    return fig
    # df2 = pd.DataFrame()
    # for i in countries: 
    #     df3 = pd.DataFrame(df1[i])
    #     df4 = pd.concat([df2, df3], axis=1)
    #     df4.columns = countries
    # fig1 = px.line(x=df4[''], 
    #                y=df4[countries],
    #                #title=f'{confirmed_cases} Total Confirmed COVID Cases',
    #                height=800,
    #                width=1200
    #                )
    # fig1.update_xaxes(title='Date')
    # fig1.update_yaxes(title='Confirmed cases')


## Question 2 Tab's layout: 
question2_layout = html.Div([
    html.Br(),
    html.H2("Plot function 'f(x) = ax2 + bx + c'", style={'textAlign': 'center'}),
    
    dcc.Graph(id='q2-plot'),
    html.Br(),
    
    html.P("Input 'a' "),
    dcc.Slider(id='a-bar', 
               min=-10, max=10,  
               step=0.5,
               value=0),
    html.Br(),
    
    html.P(" Input 'b' "),
    dcc.Slider(id='b-bar', 
               min=-10, max=10, 
               step=0.5,
               value=0,),
    html.Br(),
    
    html.P(" Input 'c' "),
    dcc.Slider(id='c-bar', 
               min=-10, max=10,
               step=0.5, 
               value=0),
    html.Br(),
])
### Question 2 Callback: 
@hw4_app.callback(
    Output(component_id='q2-plot', component_property='figure'),
    [
      Input(component_id='a-bar', component_property='value'),
      Input(component_id='b-bar', component_property='value'),
      Input(component_id='c-bar', component_property='value'),
    ]
)
def update_q2(a, b, c):
    x=np.linspace(-2, 2, 1000)
    y = a * x ** 2 + b*x + c
    data = [go.Scatter(x=x, y=y, mode='lines')]
    layout = go.Layout(title='f(x) = {}x^2 + {}x + {}'.format(a, b, c),
                       xaxis=dict(title='x'),
                       yaxis=dict(title='f(x)'))
    return {'data': data, 'layout': layout}


## Question 3 Tab's layout:
question3_layout = html.Div([
    html.Br(),
    html.H1("Calculator", style={'textAlign': 'center'}),
    html.Br(),
    
    html.Br(),
    html.P('Please enter the first number'),
    dcc.Input(id='first-number', type='value', placeholder=''),
    html.Br(),
    
    html.Br(),
    html.P('Please select one arithmetic operator'),
    dcc.Dropdown(id='q3-dropdown', options=[
            {'label':'+', 'value':'Addition'},
            {'label':'-', 'value':'Subtraction'},
            {'label':'*', 'value':'Multiplication'},
            {'label':'/', 'value':'Division'},
            {'label':'log', 'value':'Log'},
            {'label':'^', 'value':'Square'},
            {'label':'√', 'value':'Square Root'},
            ], clearable=True),
    html.Br(),
    
    html.P('Please enter the second number'),
    dcc.Input(id='second-number', type='value', placeholder=''),
    html.Br(),
    
    html.Br(),
    html.Div(id='q3-output'),
])
### Question 3 Callback: 
@hw4_app.callback(
    Output(component_id='q3-output', component_property='children'),
    [
    Input(component_id='first-number', component_property='value'),
    Input(component_id='q3-dropdown', component_property='value'),
    Input(component_id='second-number', component_property='value')
    ]
)
def update_q3(first, symbol, second):
    if symbol == 'Addition':
        result = int(first) + int(second)
        return f"The output value is {result}"
    elif symbol == 'Subtraction':
        result = int(first) - int(second)
        return f"The output value is {result}"
    elif symbol == 'Multiplication':
        result = int(first) * int(second)
        return f"The output value is {result}"
    elif symbol == 'Division':
        if int(second) == 0:
            return "Cannot divide by zero."
        else:
            result = int(first) / int(second)
            return f"The output value is {result}"
    elif symbol == 'Log': 
        result = math.log(int(first)) / math.log(int(second))
        return f"The output value is {result}"
    elif symbol == 'Square':
        result = int(first) ** int(second)
        return f"The output value is {result}"
    elif symbol == 'Square Root':
    # Root requirements: a**(1/b)
        if int(first) < 0:
            return "Cannot calculate square root of a negative number."
        else:
            result = int(first) ** (1/int(second)) 
            return f"The output value is {result}"


## Questino 4 Tab's Layout: 
question4_layout = html.Div([
    html.Br(),
    html.H1("Interactive Polynomial function", style={'textAlign': 'center'}),
    
    html.Br(),
    html.P('Please enter the Polynomial Order'),
    dcc.Input(id='poly-order', type='number', placeholder=''),
    
    html.Br(),
    dcc.Graph(id='q4-plot'),
])
### Question 4 Callback: 
@hw4_app.callback(
    Output(component_id='q4-plot', component_property='figure'),
    [Input(component_id='poly-order', component_property='value')]
)
def update_q4(poly_number):
    if not poly_number:
        return px.line(width=1200)
    x1 = np.linspace(-2, 2, 1000)
    y1 = x1**int(poly_number)
    fig2 = px.line(x=x1,
                   y=y1,
                   width=1200
                   )
    fig2.update_xaxes(title='x')
    fig2.update_yaxes(title='f(x)')
    return fig2


## Question 5 Tab's layout:
question5_layout = html.Div([
    html.Br(),
    html.H1(" Function 'f(x) = sin(x) + noise' and its Fast Fourier Transform (FFT)", style={'textAlign': 'center'}),
    html.Br(),
    
    html.Br(),
    html.P('Please enter the number of Sinusoidal-Cycle'),
    html.Br(),
    dcc.Input(id='sc', type='number', placeholder=''), # "type=" has to be "number"
    html.Br(),
    
    html.Br(),
    html.P('Please enter the Mean of the white noise'),
    html.Br(),
    dcc.Input(id='mean', type='number', placeholder=''), # "type=" has to be "number"
    html.Br(),
    
    html.Br(),
    html.P('Please enter the Standard Deviation of the white noise'),
    html.Br(),
    dcc.Input(id='sd', type='number', placeholder=''), # "type=" has to be "number"
    html.Br(),
    
    html.Br(),
    html.P('Please enter the number of samples'),
    html.Br(),
    dcc.Input(id='samples', type='number', placeholder=''), # "type=" has to be "number"
    html.Br(),
    
    html.Br(),
    dcc.Graph(id='function-plot'),
    html.Br(),
    
    html.Br(),
    html.P('The fast fourier transform FFT of above generated data'),
    html.Br(),
    
    html.Br(),
    dcc.Graph(id='fft-plot'),
])
# ### Question 5 Callback: 
@hw4_app.callback(
    [Output(component_id='function-plot', component_property='figure'),
    Output(component_id='fft-plot', component_property='figure')],
    [Input(component_id='sc', component_property='value'),
    Input(component_id='mean', component_property='value'),
    Input(component_id='sd', component_property='value'),
    Input(component_id='samples', component_property='value')]
)
def update_q5(sc, mean, sd, samples):
   x2 = np.linspace(-np.pi, np.pi, samples)
   noise = np.random.normal(mean, sd, samples)
   y2 = np.sin(sc * x2) + noise
   fft_v = np.abs(fft(y2))
   fig3 = px.line(x=x2, 
                  y=y2, 
                  width=1200)
   fig3.update_yaxes(title='f(x)')
   fig4 = px.line(x=x2, 
                  y=fft_v)
   return fig3, fig4


## Question 6 Tab's layout:
question6_layout = html.Div([
    html.Br(),
    html.H1("Two-layered neural network", style={'textAlign': 'center'}),
    
    html.Br(),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image),
                           style={'height':'40%','width':'40%', 'display':'block',
                                  'margin-left':'auto', 'margin-right':'auto'
                                  }),
    
    html.Br(),
    dcc.Graph(id='ml-plot'), 
    
    html.Br(),
    #html.P("b11"),
    dcc.Markdown(r'$b^{1}_{1}:$', mathjax=True),
    dcc.Slider(id='p1', 
               min=-10, max=10,
               marks={-10: '-10', -9: '-9', -8: '-8', 
                      -7: '-7', -6: '-6', -5:'-5', -4:'-4',
                      -3:'-3', -2:"-2", -1:"-1", 0:'0',
                      1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 
                      6:'6', 7:'7', 8:'8', 9:'9', 10:'10'},
               step=0.001, 
               value=0),
    
    html.Br(),
    #html.P("b12"),
    dcc.Markdown(r'$b^{1}_{2}:$', mathjax=True),
    dcc.Slider(id='p2', 
               min=-10, max=10,
               marks={i:f'{i}' for i in range(-10, 11)},
               # Easier to make a long Dictonary with same pattern by using a simple For-Loop
               step=0.001, 
               value=0),
    
    html.Br(),
    #html.P("b2"),
    dcc.Markdown(r'$b^{2}_{1}:$', mathjax=True),
    dcc.Slider(id='p3', 
               min=-10, max=10,
               marks={i:f'{i}' for i in range(-10, 11)},
               step=0.001, 
               value=0),
    
    html.Br(),
    #html.P("w1,1,1"),
    dcc.Markdown(r'$w^{1}_{1,1}:$', mathjax=True),
    dcc.Slider(id='p4', 
               min=-10, max=10,
               marks={i:f'{i}' for i in range(-10, 11)},
               step=0.001, 
               value=0),
    
    html.Br(),
    #html.P("w1,2,1"),
    dcc.Markdown(r'$w^{1}_{2,1}:$', mathjax=True),
    dcc.Slider(id='p5', 
               min=-10, max=10,
               marks={i:f'{i}' for i in range(-10, 11)},
               step=0.001, 
               value=0),
    
    html.Br(),
    #html.P("w2,1,1"),
    dcc.Markdown(r'$w^{2}_{1,1}:$', mathjax=True),
    dcc.Slider(id='p6', 
               min=-10, max=10,
               marks={i:f'{i}' for i in range(-10, 11)},
               step=0.001, 
               value=0),
    
    html.Br(),
    #html.P("w2,1,2"),
    dcc.Markdown(r'$w^{2}_{1,2}:$', mathjax=True),
    dcc.Slider(id='p7', 
               min=-10, max=10,
               marks={i:f'{i}' for i in range(-10, 11)},
               step=0.001, 
               value=0),
])
### Question 6 Callback: 
@hw4_app.callback(
    Output(component_id='ml-plot', component_property='figure'),
    [
     Input(component_id='p1', component_property='value'),
     Input(component_id='p2', component_property='value'),
     Input(component_id='p3', component_property='value'),
     Input(component_id='p4', component_property='value'),
     Input(component_id='p5', component_property='value'),
     Input(component_id='p6', component_property='value'),
     Input(component_id='p7', component_property='value'),
    ]
)  
def update_ml(b11, b12, b21, w111, w121, w211, w212):
    #Input to this Neural Network
    p = np.linspace(-5, 5, 1000)
    # Output of this Neural Network
    a1 = 1 / ( 1 + np.exp( - (b11*p+b12) ) )
    a2 = 1 / ( 1 + np.exp( - (w111*p+w121) ) )
    output = b21*a1 + w211*a2 + w212
    # Plot this Neural Network
    fig5 = px.line(x=p,
                   y=output
                   )
    fig5.update_xaxes(title='P')
    fig5.update_yaxes(title='a2')
    return fig5


## Final Callback:
@hw4_app.callback(
    Output(component_id='final-layout', component_property='children'),
    [Input(component_id='hw4-questions', component_property='value')]
)
def update_layout(question):
    if question == 'q1':
        return question1_layout
    elif question == 'q2':
        return question2_layout
    elif question == 'q3':
        return question3_layout
    elif question == 'q4':
        return question4_layout
    elif question == 'q5':
        return question5_layout
    elif question == 'q6':
        return question6_layout


## Create Dash-App in browser
# if __name__ == '__main__':
#     hw4_app.run_server(
#         #debug=True,
#         port=8080,
#         host='0.0.0.0')

if __name__ == '__main__':
   hw4_app.run_server(host='0.0.0.0', port=8080)


########################################## End Here ###########################################



# %%
