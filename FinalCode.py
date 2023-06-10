import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import requests
import os
import pandas as pd

# To download the job descriptions of jobs with certain job position name ( Here for example : DevOps)
url = "https://sourcestack-api.com/jobs?name=DevOps&&fields=post_full_text"
apikey = input('please insert your APi key')  # Getting API key
headers = {'x-api-key': apikey}

response = requests.get(url , headers=headers)
response.raise_for_status()

file_path = "./assets/results.json"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "wb") as file:
    file.write(response.content)

# Building ignored words list and extending it

ignored_words = list(stopwords.words('english'))
ignored_words.extend(
    '''experience working team got good goes als charts simply due 
    lives program u00e2 u20ac guidance guide society work software development u2122s grow mentality
    earliest linkedin organisation company platform sowie lifestyle linkerd 
    spent business infrastructure new tools technologies letter drive keda sexuality
     u2122re solutions u00c3 latest brightest budget '''.split())


with open('assets/results.json') as f:
    jsondata = json.load(f)
    data = jsondata['data']
    data = json.dumps(data)
    data = [data]


# CountVectorizer will only count single word. aka, 1gram word

count_vec = CountVectorizer(
    ngram_range = (1,1)
    ,stop_words = ignored_words
)

# transform the count vector result to a readable Pandas Dataframe.
tf_result = count_vec.fit_transform(data)
tf_result_df = pd.DataFrame(tf_result.toarray() ,columns=count_vec.get_feature_names_out())

#produce a Series list include keywords and its total appearance # in the text which is job descriptions.
the_sum_s = tf_result_df.sum(axis=0)
print(the_sum_s)

#Turn Series to Dataframe for easier to read and data manipulation.

the_sum_df = pd.DataFrame({
    'keyword':the_sum_s.index
    ,'tf_sum':the_sum_s.values
})

#  take words that only appear more than 15 times

the_sum_df = the_sum_df[
    the_sum_df['tf_sum']>15
    ].sort_values(by=['tf_sum'],ascending=False)

# Results!  
print(the_sum_df)