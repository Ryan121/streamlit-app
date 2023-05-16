import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
from PIL import Image


def load_lottieurl(url):
    request = requests.get(url)

    if request.status_code != 200:
        return None
    else:
        return request.json()

# Set webpage icon and tab title
# For additional page icons go to: webfx.com/tools/emoji-cheat-sheet
st.set_page_config(page_title="Example App",
                   page_icon=":mechanical_arm:",
                   layout="wide")


img_meme_man = Image.open("images/meme_man.png")


def home():

    with st.container():
        st.subheader("Hi, this site is a test project :collision:")
        st.title("It tests Streamlit - An easy to use framework")
        st.subheader("What a great introduction that was :zzz:")

    with st.container():
        st.write("---")
        
        left_col, right_col = st.columns(2)
        left_col.error('Background')
        right_col.success('Project')

        with left_col:

            st.write('##')

            st.write("""

                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
                Sollicitudin tempor id eu nisl. Aliquam vestibulum morbi blandit cursus. Egestas purus viverra accumsan in nisl nisi scelerisque eu. 
                Nec ultrices dui sapien eget mi proin sed libero. Nulla pharetra diam sit amet nisl suscipit adipiscing. Vestibulum rhoncus est pellentesque 
                elit ullamcorper dignissim cras tincidunt lobortis. Risus pretium quam vulputate dignissim suspendisse in. Pellentesque diam volutpat commodo 
                sed. Amet aliquam id diam maecenas ultricies mi eget. Nisi lacus sed viverra tellus. Pharetra massa massa ultricies mi quis. Quam elementum pulvinar 
                etiam non quam. Mi proin sed libero enim sed faucibus. Adipiscing bibendum est ultricies integer quis auctor elit. Id nibh tortor id aliquet lectus 
                proin nibh nisl.

                Nulla at volutpat diam ut venenatis. Aliquet risus feugiat in ante metus dictum at. Cursus metus aliquam eleifend mi in. Cursus in hac habitasse platea. 
                Sit amet mattis vulputate enim nulla aliquet. Elementum nisi quis eleifend quam adipiscing vitae proin sagittis nisl. Nibh venenatis cras sed felis. Id 
                consectetur purus ut faucibus pulvinar elementum integer. Ut venenatis tellus in metus vulputate eu scelerisque. Amet purus gravida quis blandit turpis cursus. 
                Id donec ultrices tincidunt arcu non sodales neque. Arcu ac tortor dignissim convallis aenean et tortor at. Porttitor rhoncus dolor purus non enim praesent 
                elementum facilisis leo. Tortor id aliquet lectus proin nibh nisl condimentum id venenatis. Risus ultricies tristique nulla aliquet enim.

                - Ut venenatis tellus in metus vulputate eu scelerisque.
                - Tortor id aliquet lectus proin nibh nisl condimentum id venenatis.
                - Potato por favor purus ut faucibus pulvinar elementum integer.
            
            """)

            st.write("[Streamlit website link > ](https://streamlit.io/)")

    # Website animation
    # https://lottiefiles.com/

    animation = "https://assets10.lottiefiles.com/packages/lf20_cHA3rG.json"

    lottie_anim = load_lottieurl(animation)

    with right_col:
        st_lottie(lottie_anim, height=400, key='image')

        st.write("""

            - Ut venenatis tellus in metus vulputate eu scelerisque.
            - Tortor id aliquet lectus proin nibh nisl condimentum id venenatis.
            - Potato por favor purus ut faucibus pulvinar elementum integer.
        
        """)


    with st.container():
        st.write("---")
        st.header("Projects")
        st.write('##')

        image_col, text_col = st.columns((1, 2))

        with image_col:
            st.image(img_meme_man)

        with text_col:
            st.subheader("How to work with streamlit")

            st.write("""

                - Ut venenatis tellus in metus vulputate eu scelerisque.
                - Tortor id aliquet lectus proin nibh nisl condimentum id venenatis.
                - Potato por favor purus ut faucibus pulvinar elementum integer.

            """)

    st.markdown("[Watch video...](https://www.youtube.com/watch?v=JwSS70SZdyM&t=288s)")

    st.write('---')
    st.write('##')

    st.subheader('Resources')

    st.write('##')  

    col1, col2, col3 = st.columns(3)

    # Image with embedded link
    with col1:
        # st.markdown("[![Foo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://www.youtube.com/watch?v=SU--XMhbWoY)")
        st.video('https://www.youtube.com/watch?v=SU--XMhbWoY')
    with col2:
        st.video('https://www.youtube.com/watch?v=VpAH2IoMzKw&t=777s')
    with col3:
        st.video('https://www.youtube.com/watch?v=XaFH2PcYI64')
    

    ex1, ex2 = st.columns(2)
    ex1.error('GitHub Source')
    ex2.success('Project')

    # Site footer
    # with container:
    #     st.title('Application footer')


# Define sections
container = st.container()


def extractdata():

    df = pd.read_csv("stroke-data-main.csv")
    # print(dataset)
    st.header('Step 1: Data Exploration')

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

    st.write('##')

    # Return filtered dataframe
    st.dataframe(df_selection)

    st.write('---')

    st.header('Step 2: Data Cleaning')

    st.subheader('Duplicate rows')
    dup_rows_df = df[df.duplicated()]
    st.write(f'{sum(dup_rows_df.count())}, duplicate rows found within the data')

    st.subheader('Null or NaN Values')
    null_values = df.isnull().sum()
    st.write(f"{null_values.count()} NaN's or nullable rows found with the data")
    
    col1, col2 = st.columns(2)
    
    # Percentage of mising BMI and Smoking values
    SmSt_mv = (df['smoking_status'].isnull().sum() / len(df['smoking_status'])*100)

    # Percentage of missing values
    BMI_mv = (df['bmi'].isnull().sum() / len(df['bmi'])*100)
    
    with col1:
        st.subheader('Smoking Status missing values:')      
    col1.error( f'{round(SmSt_mv, 1)}%', icon="🚨" ) 
    
    with col2:
        st.subheader('BMI missing values:')
    col2.warning( f'     {round(BMI_mv, 1)}%', icon="⚠️" )

    st.write('##')
    st.subheader('Histogram plot of BMI')
    col1,col2,col3 = st.columns((1,2,1))
    with col2:
        fig = plt.figure(figsize=(5,5)) # try different values
        ax = plt.axes()
        ax.hist(df['bmi'], bins = 20, color = "c", edgecolor='black')
        st.pyplot(fig)

    st.write('##')
    st.write("""
        Analysis Decisions:

        - Remove null/NaN rows
        - Remove 'Smoking Status' column all together due to the high prevelence of missing values in reference to the the total number
        - Impute BMI missing values. Use the median of the data due to the non-parametric distribution. 
    
    """)

    st.write('##')

    # Drop smoking status & impute missing BMI values (too many missing values to impute smoking status)
    new_df = df.drop(['smoking_status'], axis=1)

    st.header('New Dataframe after removing null values and smoking status data')
    st.dataframe(new_df)

    st.header('Key Stats')
    st.write(new_df.describe())
    

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
        home()


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

    current_page = st.sidebar.selectbox('Select a page', ('Home', 'Project #1'))
    
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