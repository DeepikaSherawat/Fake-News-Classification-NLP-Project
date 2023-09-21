# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:54:21 2023

@author: Deepika_Sherawat
"""

import pickle
import re
import string
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
import joblib,os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt 
import matplotlib

loaded_vectorizer = pickle.load(open('News_Detection_Tfidf_Vectorizer.sav', 'rb'))

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model




class Preprocessing:
    
    def __init__(self, data):
        self.data = data
    
    def text_preprocessing(self):
        pred_text = [self.data]
        preprocess_text = []
        lm = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        for text in pred_text:
            text = text.lower()
            text = re.sub('[^a-zA-Z0-9\s]', '', text)
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+/www\.\S+', '', text)
            text = re.sub("<.*?>+", " ", text)
            text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
            text = re.sub("\n", " ", text)
            text = re.sub("\w*\d\w*", " ", text)
            text = word_tokenize(text)
            text = [lm.lemmatize(x) for x in text if x not in stop_words]
            text = " ".join(text)
            preprocess_text.append(text)
        return preprocess_text      
    
    
class Prediction:
    
    def __init__(self, pred_data, model, vectorizer):
        self.pred_data = pred_data
        self.model = model
        self.vectorizer = vectorizer
    
    def prediction_model(self):
        preprocess_data = Preprocessing(self.pred_data).text_preprocessing()
        data = self.vectorizer.transform(preprocess_data)
        prediction = self.model.predict(data)
        
        if prediction[0] == 0:
            return "The News is Fake."
        else:
            return "The News is Real."



def main():
    
    st.title("FAKE NEWS DETECTION WEB APP")
    news_title = st.text_input("News Title :")
    news_text = st.text_input("Text:")
    all_ml_models = ["Random_Forest","Logistic_regression","Decision_tree","SVM","Naive_bayes","Gradient_boost","Ridge_classifier"]
	model_choice = st.selectbox("Select Model",all_ml_models)
    if model_choice == 'Random_Forest':
				predictor = load_prediction_models("models/newsclassifier_random_forest_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'Logistic_regression':
				predictor = load_prediction_models("models/newsclassifier_logistic_regression_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'Decision_tree':
				predictor = load_prediction_models("models/newsclassifier_decision_tree_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'SVM':
				predictor = load_prediction_models("models/newsclassifier_svm_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
            elif model_choice == 'Naive_bayes':
				predictor = load_prediction_models("models/newsclassifier_naive_bayes_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)    
            elif model_choice == 'Gradient_boost':
				predictor = load_prediction_models("models/newsclassifier_gradient_boost_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
            elif model_choice == 'Ridge_classifier':
				predictor = load_prediction_models("models/newsclassifier_ridge_classifier_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    prediction = ''
    user_data = news_title + " " + news_text
    
    if st.button('Test Result:'):
        prediction = Prediction(user_data, loaded_model, loaded_vectorizer).prediction_model()

    st.success(prediction)
    
    
if __name__ == '__main__':
   main()