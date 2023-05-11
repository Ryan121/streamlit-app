import streamlit as st
from streamlit_echarts import st_echarts



# Define sections
container = st.container()



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
        st.title('Page #2')
        st.subheader('Contents')
    
    with container:
        
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