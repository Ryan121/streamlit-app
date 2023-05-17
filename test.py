import boto3  
import os
from datasets import load_dataset
import pandas as pd
# ACCESS_KEY="AKIAV47LTA62OWKGMBPM"
# SECRET_KEY="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"

# client = boto3.client(
#     's3',
#     aws_access_key_id=ACCESS_KEY,
#     aws_secret_access_key=SECRET_KEY
# )


# client.download_file('mlops-pycaret-fastapi', 'goodreads_data.csv', 'goodreads_data.csv')


# filename = 'goodreads_data.csv'
# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# # conn = st.experimental_connection('s3', type=FilesConnection)
# # df = conn.read("mlops-pycaret-fastapi/goodreads_data.csv", input_format="csv", ttl=600)
# file_path = os.getcwd()

# print(file_path + '/' + filename)

# if not os.path.exists(file_path + '/' + filename):
#     print('hello')
#     print(os.path.exists(file_path + '\\' + filename))

dataset = load_dataset("tasksource/bigbench",'movie_recommendation')
print(dataset)

# df = pd.DataFrame(dataset["train"][:])