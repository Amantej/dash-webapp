#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from datetime import date
from dash import Dash, dcc, html, Input, Output 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from datetime import datetime
import numpy as np
from urllib.request import urlopen
import json


# In[12]:


def data_clean(df, x, county = None, state = False ):
    temp = df
    temp = temp[temp['countyFIPS'] != 0]
    if county:
        temp = temp[temp['countyFIPS'] == county]
        #print(temp)
    elif(state):
        temp = temp[temp['State'] == state]
        #print(state)
    temp1 = temp[['Date',x]]
    day_list = list(temp1[x])
    day_count = []
    day = 1
    s = False
    for i in day_list:
        if int(i) >= 1 or s == True:
            s = True
            day_count.append(day)
            day = day+1
        else:
            day_count.append(0)
    temp1['on_particular_day'] = day_count       
    y = day_count.index(1)
    temp1.drop(temp1.iloc[:y,:].index.tolist(), inplace=True)
    return temp1


# In[13]:


def regres(df, var, reg_type):
    X = np.stack(df['on_particular_day']).reshape(-1,1)
    y = np.stack(df[var]).reshape(-1,1)
    reg = LinearRegression().fit(X,y)
    predict = reg.predict(X)
    if(reg_type):
        preg = PolynomialFeatures(degree=reg_type)
        predict1 = preg.fit_transform(X)
        predict_total = LinearRegression().fit(predict1,y)
        predict = predict_total.predict(predict1)   
    reg_temp = pd.DataFrame(X)
    reg_temp1 = pd.DataFrame(y)
    reg_temp2 = pd.DataFrame(predict)
    reg_temp[1],reg_temp[2] = reg_temp1[0],reg_temp2[0]
    return reg_temp


# In[14]:


states_data2 = pd.read_csv('../state_data.csv')


# In[15]:


df_usa = states_data2.groupby(['Date']).sum().reset_index()


# In[16]:


df_1 = pd.read_csv('../norm_data')
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[17]:


app = Dash(__name__)
server = app.server

app.layout = html.Div([

    html.H1("Dashboards with Dash", style={'text-align': 'center'}),
    html.H5("Refresh, if a functionality of dash is not working", style={'text-align': 'center'}),

    html.Div([
        dcc.Dropdown(id="slct_param",
                 options=[
                     {"label": "Cases", "value": "cases"},
                     {"label": "Deaths", "value": "deaths"}],
                 multi=False,
                 value="cases",
                 style={'width': "40%"}
                 ),]),

    html.Div([dcc.DatePickerRange(
        id='my-date-picker-range',
        start_date = date(2020, 1, 23),
        min_date_allowed=date(2020, 1, 23), #2020-01-23
        max_date_allowed=date(2021, 8, 16), #2021-08-16
        initial_visible_month=date(2020, 1, 23),
        end_date=date(2021, 8, 16),
        style={'width': "40%"}
    ),]),
    
    html.Div(id='output_container', children=[]),
    html.Br(),
    
    html.Div([dcc.RadioItems(id='radio',
   options=[
       {'label': 'Normalized', 'value': 0},
       {'label': 'Log normalized', 'value': 1}
    ],
    value=0
    ),]),
    
    html.Div([dcc.Dropdown(id='state',
   options=[
{'label':'AK' , 'value':'AK' },
{'label':'AL' , 'value':'AL' },
{'label':'AR' , 'value':'AR' },
{'label':'AZ' , 'value':'AZ' },
{'label':'CA' , 'value':'CA' },
{'label':'CO' , 'value':'CO' },
{'label':'CT' , 'value':'CT' },
{'label':'DC' , 'value':'DC' },
{'label':'DE' , 'value':'DE' },
{'label':'FL' , 'value':'FL' },
{'label':'GA' , 'value':'GA' },
{'label':'HI' , 'value':'HI' },
{'label':'IA' , 'value':'IA' },
{'label':'ID' , 'value':'ID' },
{'label':'IL' , 'value':'IL' },
{'label':'IN' , 'value':'IN' },
{'label':'KS' , 'value':'KS' },
{'label':'KY' , 'value':'KY' },
{'label':'LA' , 'value':'LA' },
{'label':'MA' , 'value':'MA' },
{'label':'MD' , 'value':'MD' },
{'label':'ME' , 'value':'ME' },
{'label':'MI' , 'value':'MI' },
{'label':'MN' , 'value':'MN' },
{'label':'MO' , 'value':'MO' },
{'label':'MS' , 'value':'MS' },
{'label':'MT' , 'value':'MT' },
{'label':'NC' , 'value':'NC' },
{'label':'ND' , 'value':'ND' },
{'label':'NE' , 'value':'NE' },
{'label':'NH' , 'value':'NH' },
{'label':'NJ' , 'value':'NJ' },
{'label':'NM' , 'value':'NM' },
{'label':'NV' , 'value':'NV' },
{'label':'NY' , 'value':'NY' },
{'label':'OH' , 'value':'OH' },
{'label':'OK' , 'value':'OK' },
{'label':'OR' , 'value':'OR' },
{'label':'PA' , 'value':'PA' },
{'label':'RI' , 'value':'RI' },
{'label':'SC' , 'value':'SC' },
{'label':'SD' , 'value':'SD' },
{'label':'TN' , 'value':'TN' },
{'label':'TX' , 'value':'TX' },
{'label':'UT' , 'value':'UT' },
{'label':'VA' , 'value':'VA' },
{'label':'VT' , 'value':'VT' },
{'label':'WA' , 'value':'WA' },
{'label':'WI' , 'value':'WI' },
{'label':'WV' , 'value':'WV' },
{'label':'WY' , 'value':'WY' },],
    value = None
    ),]),
    
    html.Div([dcc.Dropdown(id='reg',
   options=[
       {'label': 'Linear', 'value': 1},
       {'label': 'Non Linear deg=2', 'value': 2},
       {'label': 'Non Linear deg=3', 'value': 3},
       {'label': 'Non Linear deg=4', 'value': 4},
       {'label': 'Non Linear deg=5', 'value': 5},
    ],
    value = None
    ),]),
    
    
     html.Div([dcc.RadioItems(id='days',
   options=[
       {'label': '7-days rolling average', 'value': 1},
    ],
    value= None
    ),]),
    
    
    html.Div([dcc.RadioItems(
        id='map',
        options=[{'label': 'Choropleth Map','value': 1},
                ],
        value= None,
        labelStyle={'display': 'inline-block'}
    ),]),
    
    #dcc.Graph(id='output-container-date-picker-range', figure={}),
    dcc.Graph(id='weekly_trend', figure={})
    #dcc.

])


# Referred to this link for choropleth map: https://plotly.com/python/mapbox-county-choropleth/ <br>
# - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html<br>
# - https://dash.plotly.com/dash-core-components

# In[18]:


@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='weekly_trend', component_property='figure')],
    [Input(component_id='slct_param', component_property='value'),
     Input(component_id='days', component_property='value'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    Input(component_id='radio', component_property='value'),
    Input(component_id='state', component_property='value'),
    Input(component_id='reg', component_property='value'),
    Input('map', 'value') ]
)
def update_graph(option_slctd,days,start_date,end_date,radio,state,reg,map):

    container = "The parameter selected: {} in date range {} to {}".format(option_slctd,start_date,end_date)
    
    if(state):
        #print(state)
        dff1 = states_data2.copy()
        mask = (dff1['Date'] > start_date) & (dff1['Date'] <= end_date)
        dff1 = dff1.loc[mask]
        dff1 =  dff1[dff1['State'] == state]
        if(option_slctd == 'cases' and radio == 0):
            fig = px.line(dff1, x ='Date', y='Normalized cases')
            if(days):
                dff1['7daysavg'] = dff1['Normalized cases'].rolling(window=7, min_periods=1).mean()
                fig1 = px.line(dff1, x ='Date', y='Normalized cases')
                fig2 = px.line(dff1, x ='Date', y='7daysavg')
                fig2.update_traces(line_color='#456987')
                fig = go.Figure(data=fig1.data + fig2.data)
            if(reg):
                #print('yuopp')
                lin = data_clean(dff1, x = 'Normalized cases')
                reg1 = regres(lin, var = 'Normalized cases', reg_type= reg)
                fig1 = px.line(reg1, x=0,y=2)
                fig2 = px.line(reg1, x=0,y=1)
                fig = go.Figure(data=fig1.data + fig2.data)
            
        elif(option_slctd == 'deaths' and radio == 0):
            fig = px.line(dff1, x ='Date', y='Normalized deaths')
            if(days):
                dff1['7daysavg'] = dff1['Normalized deaths'].rolling(window=7, min_periods=1).mean()
                fig1 = px.line(dff1, x ='Date', y='Normalized deaths')
                fig2 = px.line(dff1, x ='Date', y='7daysavg')
                fig2.update_traces(line_color='#456987')
                fig = go.Figure(data=fig1.data + fig2.data)
            if(reg):
                lin = data_clean(dff1, x = 'Normalized deaths')
                reg1 = regres(lin, var = 'Normalized deaths', reg_type= reg)
                fig1 = px.line(reg1, x=0,y=2)
                fig2 = px.line(reg1, x=0,y=1)
                fig = go.Figure(data=fig1.data + fig2.data)
        elif(option_slctd == 'cases' and radio == 1):
            fig = px.line(dff1, x ='Date', y='Log normalized cases')
            if(days):
                dff1['7daysavg'] = dff1['Log normalized cases'].rolling(window=7, min_periods=1).mean()
                fig1 = px.line(dff1, x ='Date', y='Log normalized cases')
                fig2 = px.line(dff1, x ='Date', y='7daysavg')
                fig2.update_traces(line_color='#456987')
                fig = go.Figure(data=fig1.data + fig2.data)
            if(reg):
                lin = data_clean(dff1, x = 'Log normalized cases')
                reg1 = regres(lin, var = 'Log normalized cases', reg_type= reg)
                fig1 = px.line(reg1, x=0,y=2)
                fig2 = px.line(reg1, x=0,y=1)
                fig = go.Figure(data=fig1.data + fig2.data)
        else:
            fig = px.line(dff1, x ='Date', y='Log normalized deaths')
            if(days):
                dff1['7daysavg'] = dff1['Log normalized deaths'].rolling(window=7, min_periods=1).mean()
                fig1 = px.line(dff1, x ='Date', y='Log normalized deaths')
                fig2 = px.line(dff1, x ='Date', y='7daysavg')
                fig2.update_traces(line_color='#456987')
                fig = go.Figure(data=fig1.data + fig2.data)
            if(reg):
                lin = data_clean(dff1, x = 'Log normalized deaths')
                reg1 = regres(lin, var = 'Log normalized deaths', reg_type= reg)
                fig1 = px.line(reg1, x=0,y=2)
                fig2 = px.line(reg1, x=0,y=1)
                fig = go.Figure(data=fig1.data + fig2.data)
        return container,fig
    
    
    dff = df_usa.copy()
    
    mask = (dff['Date'] > start_date) & (dff['Date'] <= end_date)
    dff = dff.loc[mask]
    print(radio)
    
    if(option_slctd == 'cases' and radio == 0):
        fig = px.line(dff, x ='Date', y='Normalized cases')
        if(days):
            dff['7daysavg'] = dff['Normalized cases'].rolling(window=7, min_periods=1).mean()
            fig1 = px.line(dff, x ='Date', y='Normalized cases')
            fig2 = px.line(dff, x ='Date', y='7daysavg')
            fig2.update_traces(line_color='#456987')
            fig = go.Figure(data=fig1.data + fig2.data)
            
        if(reg):
            lin = data_clean(dff, x = 'Normalized cases')
            reg1 = regres(lin, var = 'Normalized cases', reg_type= reg)
            fig1 = px.line(reg1, x=0,y=2)
            fig2 = px.line(reg1, x=0,y=1)
            fig = go.Figure(data=fig1.data + fig2.data)
    elif(option_slctd == 'deaths' and radio == 0):
        fig = px.line(dff, x ='Date', y='Normalized deaths')
        if(days):
            dff['7daysavg'] = dff['Normalized deaths'].rolling(window=7, min_periods=1).mean()
            fig1 = px.line(dff, x ='Date', y='Normalized deaths')
            fig2 = px.line(dff, x ='Date', y='7daysavg')
            fig2.update_traces(line_color='#456987')
            fig = go.Figure(data=fig1.data + fig2.data)
        if(reg):
            lin = data_clean(dff, x = 'Normalized deaths')
            reg1 = regres(lin, var = 'Normalized deaths', reg_type= reg)
            fig1 = px.line(reg1, x=0,y=2)
            fig2 = px.line(reg1, x=0,y=1)
            fig = go.Figure(data=fig1.data + fig2.data)
    elif(option_slctd == 'cases' and radio == 1):
        fig = px.line(dff, x ='Date', y='Log normalized cases')
        if(days):
            dff['7daysavg'] = dff['Log normalized cases'].rolling(window=7, min_periods=1).mean()
            fig1 = px.line(dff, x ='Date', y='Log normalized cases')
            fig2 = px.line(dff, x ='Date', y='7daysavg')
            fig2.update_traces(line_color='#456987')
            fig = go.Figure(data=fig1.data + fig2.data)
        if(reg):
            lin = data_clean(dff, x = 'Log normalized cases')
            reg1 = regres(lin, var = 'Log normalized cases', reg_type= reg)
            fig1 = px.line(reg1, x=0,y=2)
            fig2 = px.line(reg1, x=0,y=1)
            fig = go.Figure(data=fig1.data + fig2.data)
    else:
        fig = px.line(dff, x ='Date', y='Log normalized deaths')
        if(days):
            dff['7daysavg'] = dff['Log normalized deaths'].rolling(window=7, min_periods=1).mean()
            fig1 = px.line(dff, x ='Date', y='Log normalized deaths')
            fig2 = px.line(dff, x ='Date', y='7daysavg')
            fig2.update_traces(line_color='#456987')
            fig = go.Figure(data=fig1.data + fig2.data)
        if(reg):
            lin = data_clean(dff, x = 'Log normalized deaths')
            reg1 = regres(lin, var = 'Log normalized deaths', reg_type= reg)
            fig1 = px.line(reg1, x=0,y=2)
            fig2 = px.line(reg1, x=0,y=1)
            fig = go.Figure(data=fig1.data + fig2.data)
    if(map):
        if(option_slctd=='cases'): 
            map = 'Cases'
        else:
            map = 'Death'
        fig = px.choropleth_mapbox(df_1, geojson=counties, locations='countyFIPS', color=map,
                           color_continuous_scale="blues",
                           range_color=[df_1[map].min(), df_1[map].max()],
                           mapbox_style="open-street-map",
                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           hover_data=['County Name', 'StateFIPS', map, 'Population']
                          )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600,)


    return container, fig


# In[19]:


#app.run_server(debug= True,port=8055)


# In[20]:


#pip freeze>requirements.txt


# In[ ]:




