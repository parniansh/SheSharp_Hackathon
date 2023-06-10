# SheSharp_Hackathon

# Keyword Extractor From Job Descriptions

The goal of this program is to extract the keywords from all of the job descreptions for a certain job position ( for instance  : DevOps),  
so the job seeker would know what keywords to use in his or her resume.
This way we can increase the chance of the resume pass through bots and have more chance of getting the interviews.


This application is written in python. I have used pandas and sklearn libraries in order to extract keywords.
I have used an unsupervised procedure so there is no pre-request training data and we capture new words and phrases automatically.

For learning purposes TF_IDF algorithm has been used to  evaluate how relevant a word is to a document in a collection of documents.
Based on TF-IDF, those unique and important words should have high TF-IDF values in a certain document.

The code is well-commented so enjoy!
