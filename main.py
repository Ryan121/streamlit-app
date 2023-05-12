import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import matplotlib.pyplot as plt

# Set webpage icon and tab title
# For additional page icons go to: webfx.com/tools/emoji-cheat-sheet
st.set_page_config(page_title="Example App",
                   page_icon=":mechanical_arm:",
                   layout="wide")

# Define sections
container = st.container()


def extractdata():

    df = pd.read_csv("stroke-data-main.csv")
    # print(dataset)
    st.header('Raw Dataframe')

    with open('styles.css') as cs:
        st.markdown(f'<style> { cs.read() } </style>', unsafe_allow_html=True)
    
    st.subheader('Filter data here:')

    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    
    with col1:
        gender = st.multiselect("Select a gender:",
                                options=df["gender"].unique(),
                                default=df["gender"].unique())
    
    with col2:
        heart_disease = st.multiselect("Select if an individual had heart disease:",
                                options=df["heart_disease"].unique(),
                                default=df["heart_disease"].unique())
        
    with col3:
        ever_married = st.multiselect("Select if an individual was ever married:",
                                options=df["ever_married"].unique(),
                                default=df["ever_married"].unique())
    
    with col4:
        work_type = st.multiselect("Select an individuals type of work:",
                                options=df["work_type"].unique(),
                                default=df["work_type"].unique())
    
    with col5:
        hypertension = st.multiselect("Select if an individual had hypertension:",
                                options=df["hypertension"].unique(),
                                default=df["hypertension"].unique())

    with col6:
        Residence_type = st.multiselect("Select the residence type:",
                                options=df["Residence_type"].unique(),
                                default=df["Residence_type"].unique())

    with col7:
        smoking_status = st.multiselect("Select if an individual smokes:",
                                options=df["smoking_status"].unique(),
                                default=df["smoking_status"].unique())

    with col8:
        stroke = st.multiselect("Define if an individual has had a stroke:",
                                options=df["stroke"].unique(),
                                default=df["stroke"].unique())
    

    df_selection = df.query(
        "stroke == @stroke & smoking_status == @smoking_status & Residence_type == @Residence_type & work_type == @work_type & ever_married == @ever_married & heart_disease == @heart_disease & hypertension == @hypertension & gender == @gender"
    )

    # Return filtered dataframe
    st.dataframe(df_selection)

    st.title('Data Exploration')

    st.header('Duplicate rows')
    dup_rows_df = df[df.duplicated()]
    st.subheader(dup_rows_df)

    st.header('Null Values')
    null_values = df.isnull().sum()
    st.subheader(null_values)
    
    col1, col2 = st.columns(2)

    with col1:
        # Percentage of mising BMI and Smoking values
        SmSt_mv = (df['smoking_status'].isnull().sum() / len(df['smoking_status'])*100)
        st.header('Smoking Status missing values:')
        st.subheader(f'{round(SmSt_mv, 1)}%')

    
    with col2:
        # Percentage of missing values
        BMI_mv = (df['bmi'].isnull().sum() / len(df['bmi'])*100)
        st.header('BMI missing values:')
        st.subheader(f'{round(BMI_mv, 1)}%')

    # Drop smoking status & impute missing BMI values (too many missing values to impute smoking status)
    new_df = df.drop(['smoking_status'], axis=1)

    st.header('New Dataframe after removing smoking status data')
    st.subheader('This category was removed due to too many missing values')
    st.dataframe(new_df)

    st.header('Key Stats')
    st.write(new_df.describe())
    
    fig = plt.figure(figsize=(5,5)) # try different values
    ax = plt.axes()
    ax.hist(df['bmi'], bins = 20, color = "c", edgecolor='black')
    st.pyplot(fig)
    

    # Impute missing BMI level rows with BMI median
    new_df['bmi'] = new_df['bmi'].fillna(new_df['bmi'].median())
    
    option = {
        "title": [
            {"text": "Dataset BMI", "left": "center"},
            {
                "text": "upper: Q3 + 1.5 * IQR \nlower: Q1 - 1.5 * IQR",
                "borderColor": "#999",
                "borderWidth": 1,
                "textStyle": {"fontWeight": "normal", "fontSize": 14, "lineHeight": 20},
                "left": "10%",
                "top": "90%",
            },
        ],
        "dataset": [
            {
                "source": [
                    
                        new_df['bmi'].values.tolist()
                       
                ]
            },
            {
                "transform": {
                    "type": "boxplot",
                    "config": {"itemNameFormatter": "expr {value}"},
                }
            },
            {"fromDatasetIndex": 1, "fromTransformResult": 1},
        ],
        "tooltip": {"trigger": "item", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "10%", "right": "10%", "bottom": "15%"},
        "yAxis": {
            "type": "category",
            "boundaryGap": True,
            "nameGap": 30,
            "splitArea": {"show": False},
            "splitLine": {"show": False},
        },
        "xAxis": {
            "type": "value",
            "name": "BMI",
            "splitArea": {"show": True},
        },
        "series": [
            {"name": "boxplot", "type": "boxplot", "datasetIndex": 1},
            {"name": "outlier", "type": "scatter", "datasetIndex": 2},
        ],
    }
    st_echarts(option, height="500px")





def homepage():
    # Site header
    with container:
        st.title('Welcome to this dev streamlit app')
        st.subheader('Contents')

        # Image with embedded link
        st.markdown("[![Foo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://www.youtube.com/watch?v=SU--XMhbWoY)")

        # Video
        st.video('https://www.youtube.com/watch?v=SU--XMhbWoY')

        ex1, ex2 = st.columns(2)
        ex1.error('GitHub Source')
        ex2.success('Project')

    # Site footer
    with container:
        st.title('Application footer')


def projectpage():
   # Site header
    with container:
        st.title('Project #1 - Stroke Predicition in Adults')
    
    with container:

        extractdata()

        
        option = {
            "legend": {"top": "bottom"},
            "toolbox": {
                "show": True,
                "feature": {
                    "mark": {"show": True},
                    "dataView": {"show": True, "readOnly": False},
                    "restore": {"show": True},
                    "saveAsImage": {"show": True},
                },
            },
            "series": [
                {
                    "name": "面积模式",
                    "type": "pie",
                    "radius": [50, 250],
                    "center": ["50%", "50%"],
                    "roseType": "area",
                    "itemStyle": {"borderRadius": 8},
                    "data": [
                        {"value": 40, "name": "rose 1"},
                        {"value": 38, "name": "rose 2"},
                        {"value": 32, "name": "rose 3"},
                        {"value": 30, "name": "rose 4"},
                        {"value": 28, "name": "rose 5"},
                        {"value": 26, "name": "rose 6"},
                        {"value": 22, "name": "rose 7"},
                        {"value": 18, "name": "rose 8"},
                    ],
                }
            ],
        }
        st_echarts(
            options=option, height="600px",
        )

    # Site footer
    with container:
        st.title('Application footer')

        liquidfill_option = {
            "series": [{"type": "liquidFill", "data": [0.6, 0.5, 0.4, 0.3]}]
        }
        st_echarts(liquidfill_option)


def navigation():

    current_page = st.sidebar.selectbox('Select a page', ('Home', 'Project #1', 'Project #2', 'Project #3'))
    
    if current_page == 'Home':
        homepage()

    elif current_page == 'Project #1':
        projectpage()

    

def main():
    st.sidebar.title('Menu')
    st.sidebar.subheader('Page Navigator')

    navigation()


# Start Application
if __name__ == '__main__':
    main()