import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# To address the imbalanced daatset problem we first over sample the minoirty class
from imblearn.over_sampling import RandomOverSampler

# https://plotly.com/python/plotly-express/
import plotly.express as px  # pip install plotly-express


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

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


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

    st.write('##')

    st.write('---')

    st.header(":bar_chart: KPI's")

    average_glucose = round(df_selection["avg_glucose_level"].median(), 1)
    average_bmi = round(df_selection["bmi"].median(), 1)
    average_age = round(df_selection["age"].median())

    stroke_vals = df_selection["stroke"].value_counts()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Average Glucose:')
        st.subheader(f' {average_glucose} mg/dL')
    
    with col2:
        st.subheader('Average BMI:')
        st.subheader(f' {average_bmi}')
    with col3:
        st.subheader('Average Age:')
        st.subheader(f' {average_age}')
    
    col1, col2, col3 = st.columns(3)

    non_stroke = round( (stroke_vals[0] / stroke_vals.sum()) * 100 , 1)
    y_stroke = round( (stroke_vals[1] / stroke_vals.sum()) * 100 , 1)

    with col1:
        st.subheader('Ratio of Non-stroke : Stroke')
        col1_1, col2_1 = st.columns(2)
        with col1_1:
            st.subheader(f' { non_stroke  }% ')
        with col2_1:
            st.subheader(f' { y_stroke }%')
        
    with col2:
        st.subheader('Data Imbalance:')
        st.subheader(f' { round( abs(non_stroke - y_stroke ), 1)  }%')
    with col3:
        st.subheader('Dataset Size:')
        st.subheader(f' { stroke_vals.sum() } records')

    st.write('---')

    st.header('Step 2: Data Cleaning')

    st.subheader('Duplicate rows')
    dup_rows_df = df_selection[df_selection.duplicated()]
    st.write(f'{sum(dup_rows_df.count())}, duplicate rows found within the data')

    st.subheader('Null or NaN Values')
    null_values = df_selection.isnull().sum()
    st.write(f"{null_values.count()} NaN's or nullable rows found with the data")
    
    col1, col2 = st.columns(2)
    
    # Percentage of mising BMI and Smoking values
    SmSt_mv = (df_selection['smoking_status'].isnull().sum() / len(df_selection['smoking_status'])*100)

    # Percentage of missing values
    BMI_mv = (df_selection['bmi'].isnull().sum() / len(df_selection['bmi'])*100)
    
    with col1:
        st.subheader('Smoking Status missing values:')      
    col1.error( f'{round(SmSt_mv, 1)}%', icon="🚨" ) 
    
    with col2:
        st.subheader('BMI missing values:')
    col2.warning( f'     {round(BMI_mv, 1)}%', icon="⚠️" )

    st.write('##')

    # Plot 1
    fig_bmi_hist = px.histogram(
    df_selection, 
    x="bmi",
    nbins=100,
    title="<b>Histogram of Adult BMI</b>",
    template="plotly_white",
    )
    
    fig_bmi_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # Plot 2
    fig_bmi_gen_hist = px.histogram(
    df_selection, 
    x="bmi",
    color="gender", 
    nbins=100,
    title="<b>Histogram of Male, Female & Other BMI</b>",
    template="plotly_white",
    )
    
    fig_bmi_gen_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )


    # Plot 3
    bmi_by_work_type = df_selection.groupby(by=["work_type"])[["bmi"]].median()

    # bmi_by_work_type1 = df_selection.groupby(by=["work_type"]).sum()[["bmi"]]

    bmi_by_work_type_bar = px.bar(
        bmi_by_work_type,
        y=bmi_by_work_type.index,
        x="bmi",
        orientation="h",
        title="<b>BMI by type of work</b>",
        template="plotly_white",
    )
    bmi_by_work_type_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )
    
    # col1,col2,col3 = st.columns((1,2,1))
    col1,col2, col3 = st.columns(3)
    col1.plotly_chart(fig_bmi_hist, use_container_width=True)
    col2.plotly_chart(fig_bmi_gen_hist, use_container_width=True)
    col3.plotly_chart(bmi_by_work_type_bar, use_container_width=True)


    fig_bmi_boxplot = px.box(
        df_selection,
        # Use x instead of y argument for horizontal plot
        x="bmi",
        points="all",
        notched=True,
        title="<b>BMI Boxplot</b>",
        template="plotly_white",
        
    )
    fig_bmi_boxplot.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    st.plotly_chart(fig_bmi_boxplot, use_container_width=True)
    # fig = plt.figure(figsize=(5,5)) # try different values
    # ax = plt.axes()
    # ax.hist(df['bmi'], bins = 20, color = "c", edgecolor='black')
    # st.pyplot(fig)

    st.write('##')
    st.write("""
        Analysis Decisions:

        - Remove null/NaN rows
        - Remove 'Smoking Status' column all together due to the high prevelence of missing values in reference to the the total number
        - Impute BMI missing values. Use the median of the data due to the non-parametric distribution. 
    
    """)

    st.write('##')

    # Drop smoking status & impute missing BMI values (too many missing values to impute smoking status)
    new_df = df_selection.drop(['smoking_status'], axis=1)

    # Impute missing BMI level rows with BMI median
    new_df['bmi'] = new_df['bmi'].fillna(new_df['bmi'].median())

    st.header('New Dataframe after removing null values and smoking status data & imputing BMI data')
    st.write('##')
    st.dataframe(new_df)

    st.write('##')

    st.write('---')



    st.header('Key Stats')
    st.write(new_df.describe())
    

    st.write('##')

    st.write('---')

    # Convert categorical data into numerical data - Binary categorical fields were re-labelled as 0's and 1's &
    # fields with more than 2 unique values were labelled with one hot encoding

    # Binary/nominal variables - ever_married, Residence_type
    # categorical - One hot variables - work_type, smoking_status, gender

    # transform nominal variables that only have 2 values
    class_mapping = {label: idx for idx, label in enumerate(np.unique(new_df['ever_married']))}
    # print(class_mapping)
    new_df['ever_married'] = new_df['ever_married'].map(class_mapping)

    class_mapping = {label: idx for idx, label in enumerate(np.unique(new_df['Residence_type']))}
    # print(class_mapping)
    new_df['Residence_type'] = new_df['Residence_type'].map(class_mapping)

    class_mapping = {label: idx for idx, label in enumerate(np.unique(new_df['gender']))}
    # print(class_mapping)
    new_df['gender'] = new_df['gender'].map(class_mapping)

    # transform nominal variables that have more than 2 values
    new_df[['work_type']] = new_df[['work_type']].astype(str)

    # concatenate nominal variables from pd.getdummies &
    transpose = pd.get_dummies(new_df[['work_type']])

    # And the ordinal variables to form the final dataset
    new_df = pd.concat([new_df,transpose], axis=1)[['id','age','hypertension','heart_disease','ever_married','Residence_type',
                                            'avg_glucose_level','bmi','gender','work_type_Govt_job','work_type_Never_worked',
                                            'work_type_Private','work_type_children','work_type_Self-employed','stroke']]
    
    st.header('Data Conversion for ML')
    st.write("""

        - Convert categorical data into numerical data - Binary categorical fields were re-labelled as 0's and 1's & 
        fields with more than 2 unique values were labelled with one hot encoding
        - Binary/nominal variables - ever_married, Residence_type
        - Categorical - One hot variables - work_type, smoking_status, gender
    
    """)
    st.write('##')

    st.subheader('Filter data here:')
    st.write('##')
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    col9, col10, col11 = st.columns(3)
    col12, col13 =  st.columns(2)
    
    with col1:
        age_choice = st.selectbox(
            'Include age parameter?',
            ('Yes', 'No'))
        if str(age_choice) == 'No':
            new_df = new_df.drop('age', axis=1)
    with col2:
        hyp_choice = st.selectbox(
            'Include hypertension parameter?',
            ('Yes', 'No'))
        if str(hyp_choice) == 'No':
            new_df = new_df.drop('hypertension', axis=1)
    with col3:
        heart_disease_choice = st.selectbox(
            'Include heart disease parameter?',
            ('Yes', 'No'))
        if str(heart_disease_choice) == 'No':
            new_df = new_df.drop('heart_disease', axis=1)
    with col4:
        ever_married_choice = st.selectbox(
            'Include ever married parameter?',
            ('Yes', 'No'))
        if str(ever_married_choice) == 'No':
            new_df = new_df.drop('ever_married', axis=1)
    with col5:
        Residence_type_choice = st.selectbox(
            'Include Residence type parameter?',
            ('Yes', 'No'))
        if str(Residence_type_choice) == 'No':
            new_df = new_df.drop('Residence_type', axis=1)
    with col6:
        av_glucose_level_choice = st.selectbox(
            'Include average glucose level parameter?',
            ('Yes', 'No'))
        if str(av_glucose_level_choice) == 'No':
            new_df = new_df.drop('avg_glucose_level', axis=1)
    with col7:
        bmi_choice = st.selectbox(
            'Include BMI parameter?',
            ('Yes', 'No'))
        if str(bmi_choice) == 'No':
            new_df = new_df.drop('bmi', axis=1)
    with col8:
        gender_choice = st.selectbox(
            'Include gender parameter?',
            ('Yes', 'No'))
        if str(gender_choice) == 'No':
            new_df = new_df.drop('gender', axis=1)
    with col9:
        work_type_Govt_job_choice = st.selectbox(
            'Include work type govt job parameter?',
            ('Yes', 'No'))
        if str(work_type_Govt_job_choice) == 'No':
            new_df = new_df.drop('work_type_Govt_job', axis=1)
    with col10:
        work_type_Never_worked_choice = st.selectbox(
            'Include work type never worked parameter?',
            ('Yes', 'No'))
        if str(work_type_Never_worked_choice) == 'No':
            new_df = new_df.drop('work_type_Never_worked', axis=1)
    with col11:
        work_type_Private_choice = st.selectbox(
            'Include work type private parameter?',
            ('Yes', 'No'))
        if str(work_type_Private_choice) == 'No':
            new_df = new_df.drop('work_type_Private', axis=1)
    with col12:
        work_type_children_choice = st.selectbox(
            'Include work type children parameter?',
            ('Yes', 'No'))
        if str(work_type_children_choice) == 'No':
            new_df = new_df.drop('work_type_children', axis=1)
    with col13:
        work_type_Self_employed = st.selectbox(
            'Include work type self employed parameter?',
            ('Yes', 'No'))
        if str(work_type_Self_employed) == 'No':
            new_df = new_df.drop('work_type_Self-employed', axis=1)

    

    st.write('##') 
    st.dataframe(new_df)
    st.write('##')

    # Define label vector
    y = new_df[['stroke']]

    # Define feature array
    X = new_df.drop(['id','stroke'], axis=1)

    # Define random forest model
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(X, y)

    # Get importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame(importance)

    # Summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

    
    # Plot
    importance_plt_bar = px.bar(
        importance_df,
        y=X.columns,
        x=importance_df[0],
        orientation="h",
        title="<b>Feature Importance</b>",
        template="plotly_white",
    )
    importance_plt_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    st.plotly_chart(importance_plt_bar, use_container_width=True)


    # Standardize data
    ss = StandardScaler()

    X = ss.fit_transform(X)

    st.write('##') 
    st.write('---') 

    st.subheader('Data Imbalance Issue')
    st.write("""

        Addressing the data imbalenced problem in the dataset:

        - Currently the ouput label 'stroke' has a 98.2:1.8 ratio of non-stroke:stroke events which will negatively bias
          any model trained using the dataset as is. Rebalancing the dataset must be performed before using the data to 
          train any models
        
        - Data rebalanced with RandomOverSampler

    """)

    # Instantiate oversampler 
    rs = RandomOverSampler()

    # Run model on dataset
    X, y = rs.fit_resample(X, y)

    yy = pd.DataFrame(y)

    stroke_vals = yy.value_counts()
    
    col1, col2, col3 = st.columns(3)

    non_stroke = round( (stroke_vals[0] / stroke_vals.sum()) * 100 , 1)
    y_stroke = round( (stroke_vals[1] / stroke_vals.sum()) * 100 , 1)

    with col1:
        st.subheader('Ratio of Non-stroke : Stroke')
        col1_1, col2_1 = st.columns(2)
        with col1_1:
            st.subheader(f' { non_stroke  }% ')
        with col2_1:
            st.subheader(f' { y_stroke }%')
        
    with col2:
        st.subheader('Data Imbalance:')
        st.subheader(f' { round( abs(non_stroke - y_stroke ), 1)  }%')
    with col3:
        st.subheader('Dataset Size:')
        st.subheader(f' { stroke_vals.sum() } records')

    st.write('---')


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

    # Instantiate model
    rf = RandomForestClassifier()

    # Train model
    rf.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf.predict(X_test)

    # Define scoring function
    def classification_eval(y_test, y_pred):
        res1 = f'accuracy  = {np.round(accuracy_score(y_test, y_pred), 3)}'
        res2 = f'precision = {np.round(precision_score(y_test, y_pred), 3)}'
        res3 = f'recall    = {np.round(recall_score(y_test, y_pred), 3)}'
        res4 = f'f1-score  = {np.round(f1_score(y_test, y_pred), 3)}'
        res5 = f'roc auc   = {np.round(roc_auc_score(y_test, y_pred), 3)}'
        print(res1)
        print(res2)
        print(res3)
        print(res4)
        print(res5)
        
        return [res1, res2, res3, res4, res5]

    # Assess accuracy
    results = classification_eval(y_test, y_pred)

    st.write('##')

    st.subheader('Stroke Classification Results')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.subheader(results[0])
     
    with col2:
        st.subheader(results[1])

    with col3:
        st.subheader(results[2])

    with col4:
        st.subheader(results[3])

    with col5:
        st.subheader(results[4])

    st.write('---')
    
    # option = {
    #     "title": [
    #         {"text": "Dataset BMI", "left": "center"},
    #         {
    #             "text": "upper: Q3 + 1.5 * IQR \nlower: Q1 - 1.5 * IQR",
    #             "borderColor": "#999",
    #             "borderWidth": 1,
    #             "textStyle": {"fontWeight": "normal", "fontSize": 14, "lineHeight": 20},
    #             "left": "10%",
    #             "top": "90%",
    #         },
    #     ],
    #     "dataset": [
    #         {
    #             "source": [
                    
    #                     new_df['bmi'].values.tolist()
                       
    #             ]
    #         },
    #         {
    #             "transform": {
    #                 "type": "boxplot",
    #                 "config": {"itemNameFormatter": "expr {value}"},
    #             }
    #         },
    #         {"fromDatasetIndex": 1, "fromTransformResult": 1},
    #     ],
    #     "tooltip": {"trigger": "item", "axisPointer": {"type": "shadow"}},
    #     "grid": {"left": "10%", "right": "10%", "bottom": "15%"},
    #     "yAxis": {
    #         "type": "category",
    #         "boundaryGap": True,
    #         "nameGap": 30,
    #         "splitArea": {"show": False},
    #         "splitLine": {"show": False},
    #     },
    #     "xAxis": {
    #         "type": "value",
    #         "name": "BMI",
    #         "splitArea": {"show": True},
    #     },
    #     "series": [
    #         {"name": "boxplot", "type": "boxplot", "datasetIndex": 1},
    #         {"name": "outlier", "type": "scatter", "datasetIndex": 2},
    #     ],
    # }
    # st_echarts(option, height="500px")





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

        
        # option = {
        #     "legend": {"top": "bottom"},
        #     "toolbox": {
        #         "show": True,
        #         "feature": {
        #             "mark": {"show": True},
        #             "dataView": {"show": True, "readOnly": False},
        #             "restore": {"show": True},
        #             "saveAsImage": {"show": True},
        #         },
        #     },
        #     "series": [
        #         {
        #             "name": "面积模式",
        #             "type": "pie",
        #             "radius": [50, 250],
        #             "center": ["50%", "50%"],
        #             "roseType": "area",
        #             "itemStyle": {"borderRadius": 8},
        #             "data": [
        #                 {"value": 40, "name": "rose 1"},
        #                 {"value": 38, "name": "rose 2"},
        #                 {"value": 32, "name": "rose 3"},
        #                 {"value": 30, "name": "rose 4"},
        #                 {"value": 28, "name": "rose 5"},
        #                 {"value": 26, "name": "rose 6"},
        #                 {"value": 22, "name": "rose 7"},
        #                 {"value": 18, "name": "rose 8"},
        #             ],
        #         }
        #     ],
        # }
        # st_echarts(
        #     options=option, height="600px",
        # )

    # Site footer
    with container:
        # st.title('Application footer')

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