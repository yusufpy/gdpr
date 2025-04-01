
t='''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Data
data = pd.read_csv('gdpr_violations.csv')
additional_data=pd.read_csv('gdpr_violations_extended.csv')
data=data[['article_violated','summary']]
data=pd.concat([data,additional_data])

# Encode 'article_violated' using LabelEncoder
article_encoder = LabelEncoder()
data["article_violated_encoded"] = article_encoder.fit_transform(data["article_violated"])

# Risk Assessment Function
def risk_assessment(article):
    high_risk = {
        "Art. 5 GDPR", "Art. 6 GDPR", "Art. 7 GDPR", "Art. 9 GDPR", "Art. 10 GDPR", 
        "Art. 17 GDPR", "Art. 22 GDPR", "Art. 33 GDPR", "Art. 44 GDPR", "Art. 45 GDPR", 
        "Art. 46 GDPR", "Art. 83 GDPR"
    }
    
    medium_risk = {
        "Art. 8 GDPR", "Art. 12 GDPR", "Art. 13 GDPR", "Art. 14 GDPR", "Art. 15 GDPR", 
        "Art. 18 GDPR", "Art. 19 GDPR", "Art. 20 GDPR", "Art. 23 GDPR", "Art. 35 GDPR"
    }
    
    low_risk = {
        "Art. 11 GDPR", "Art. 16 GDPR", "Art. 21 GDPR", "Art. 37 GDPR", "Art. 43 GDPR"
    }
    
    if article in high_risk:
        return "High"
    elif article in medium_risk:
        return "Medium"
    elif article in low_risk:
        return "Low"
    else:
        return "Unknown" 

data["risk_level"] = data["article_violated"].apply(risk_assessment)

# Encode `risk_level`
risk_encoder = LabelEncoder()
data["risk_level_encoded"] = risk_encoder.fit_transform(data["risk_level"])

# Feature Engineering
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data["summary"])

# Train-Test Split
X_train, X_test, y_train_articles, y_test_articles, y_train_risk, y_test_risk = train_test_split(
    X_text, data["article_violated_encoded"], data["risk_level_encoded"], test_size=0.2, random_state=42
)

# Train Models
article_model = RandomForestClassifier(n_estimators=100, random_state=42)
article_model.fit(X_train, y_train_articles)

risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_train_risk)

# Streamlit App
st.title("GDPR Violation Risk Assessment")
st.write("Enter a GDPR violation summary to check the likely violated article and risk level.")

user_input = st.text_area("Enter GDPR Violation Summary:")

if st.button("Check"):
    if user_input:
        user_input_vectorized = vectorizer.transform([user_input])
        predicted_article = article_model.predict(user_input_vectorized)
        predicted_risk = risk_model.predict(user_input_vectorized)
        
        predicted_article_label = article_encoder.inverse_transform(predicted_article)[0]
        predicted_risk_label = risk_encoder.inverse_transform(predicted_risk)[0]
        
        st.subheader("Prediction Results")
        st.write(f"**Likely Violated Article:** {predicted_article_label}")
        st.write(f"**Risk Level:** {predicted_risk_label}")
    else:
        st.warning("Please enter a violation summary.")
'''


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('gdpr_violations.csv')
additional_data=pd.read_csv('/content/gdpr_violations_additional.csv')
data=data[['article_violated','summary']]
data=pd.concat([data,additional_data])

data['article_violated'] = data['article_violated'].str.replace(r'\(1\)', '', regex=True)
data['article_violated'] = data['article_violated'].str.replace(r'\(2\)', '', regex=True)
data['article_violated'] = data['article_violated'].str.replace(r'\(3\)', '', regex=True)
data['article_violated'] = data['article_violated'].str.replace(r'\(4\)', '', regex=True)
data['article_violated'] = data['article_violated'].str.replace(r'\(5\)', '', regex=True)

# Encode 'article_violated' using LabelEncoder
article_encoder = LabelEncoder()
data["article_violated_encoded"] = article_encoder.fit_transform(data["article_violated"])

# Risk Assessment Function
def risk_assessment(article):
    high_risk = {
        "Art. 5 GDPR", "Art. 6 GDPR", "Art. 7 GDPR", "Art. 9 GDPR", "Art. 10 GDPR",
        "Art. 17 GDPR", "Art. 22 GDPR", "Art. 33 GDPR", "Art. 44 GDPR", "Art. 45 GDPR",
        "Art. 46 GDPR", "Art. 83 GDPR"
    }

    medium_risk = {
        "Art. 8 GDPR", "Art. 12 GDPR", "Art. 13 GDPR", "Art. 14 GDPR", "Art. 15 GDPR",
        "Art. 18 GDPR", "Art. 19 GDPR", "Art. 20 GDPR", "Art. 23 GDPR", "Art. 35 GDPR"
    }

    low_risk = {
        "Art. 11 GDPR", "Art. 16 GDPR", "Art. 21 GDPR", "Art. 37 GDPR", "Art. 43 GDPR"
    }

    if article in high_risk:
        return "High"
    elif article in medium_risk:
        return "Medium"
    elif article in low_risk:
        return "Low"
    else:
        return "High"  # If the article is not in the predefined list


data["risk_level"] = data["article_violated"].apply(risk_assessment)

# Encode `risk_level`
risk_encoder = LabelEncoder()
data["risk_level_encoded"] = risk_encoder.fit_transform(data["risk_level"])

# Feature Engineering
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(data["summary"])

# Train-Test Split
X_train, X_test, y_train_articles, y_test_articles, y_train_risk, y_test_risk = train_test_split(
    X_text, data["article_violated_encoded"], data["risk_level_encoded"], test_size=0.2, random_state=42
)

# Train Models
article_model = RandomForestClassifier(n_estimators=100, random_state=42)
article_model.fit(X_train, y_train_articles)

risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_train_risk)

# Streamlit App
st.title("GDPR Violation Risk Assessment")
st.write("Enter a GDPR violation summary to check the likely violated article and risk level.")

user_input = st.text_area("Enter GDPR Violation Summary:")

if st.button("Check"):
    if user_input:
        user_input_vectorized = vectorizer.transform([user_input])
        predicted_article = article_model.predict(user_input_vectorized)
        predicted_risk = risk_model.predict(user_input_vectorized)
        
        predicted_article_label = article_encoder.inverse_transform(predicted_article)[0]
        predicted_risk_label = risk_encoder.inverse_transform(predicted_risk)[0]
        
        st.subheader("Prediction Results")
        st.write(f"**Likely Violated Article:** {predicted_article_label}")
        st.write(f"**Risk Level:** {predicted_risk_label}")
    else:
        st.warning("Please enter a violation summary.")
