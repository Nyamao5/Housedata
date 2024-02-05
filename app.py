import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import openpyxl
import streamlit as st

df = pd.read_excel('HouseData (1).xlsx')

#add image of the organisation / logo

#st.image("Africdsa.jpeg")

#add the title to our app

st.title("Linear Regression App For Prediction of House Price ")

#Add the header

st.header("Dataset Concept.", divider="rainbow")


# add our paragraph explaining the dataset

st.write("""House prices in a particular location can be influenced by several key factors, 
         including the condition of the property, the number of bathrooms and bedrooms, the specific neighborhood or location,
         and the associated zip code. The condition of a house is a crucial determinant, as well-maintained properties or those with recent renovations often command higher prices.
         The number of bathrooms and bedrooms also plays a significant role, catering to different household needs and preferences. 
         The location and neighborhood characteristics, such as proximity to amenities, schools, and safety, contribute to the overall desirability and, consequently,
         the pricing of houses. Additionally, the zip code serves as a geographic identifier, reflecting broader regional factors that can impact housing values,
         such as local economic conditions and community infrastructure.
         Together, these factors create a nuanced landscape for understanding and assessing house prices in a specific location.""")



#------------------------------------------- dispaly our EDA --------------------------------

st.header("Exploratory Data Analysis (EDA.)", divider="rainbow")


if st.checkbox("Dataset info"):
    st.write("Dataset info",df.info())


if st.checkbox("Number of Rows"):
    st.write("Number of rows:", df.shape[0])

if st.checkbox("Number of Columns"):
    st.write("Number of columns:", df.shape[1])

if st.checkbox("Column names"):
    st.write("Column names:", df.columns.tolist())
    
if st.checkbox("Data types"):    
    st.write("Data types:", df.dtypes)
    
if st.checkbox("Missing Values"):
    st.write("Missing values:", df.isnull().sum())

if st.checkbox("Statistical Summary"):
    st.write("Statistical Summary:", df.describe())
    

#--------------------------------- Visualisation ------------------------------

st.header("Visualization of the Dataset (VIZ)",  divider="rainbow")
 
 
# bar chart

if st.checkbox("Bar chart of Inflation rate Against GDP"):
    st.write("Bar chart of Inflation rate Against GDP")
    st.bar_chart(x= "bedrooms" , y = "price" , data =df , color=["#FF0000"])    


# Create the Bar chart
if st.checkbox("Bar Chart for Bathrooms Against Price"):
    st.write("Bar Chart for Bathrooms Against Price")
    st.bar_chart(x="bathrooms",  y="price",data=df , color=["#FF0000"])


# create a Line chart

if st.checkbox("Line Chart for Grade Against Price"):
    st.write("Line Chart for Grade Against Price")
    st.line_chart(x="grade",y="price", data=df , color=["#ffaa0088"])


# create a Scatter

if st.checkbox("Scatter Chart of Floors Against Price"):
    st.write("Scatter Chart of Floors Against Price")
    # Create the histogram using Altair
    st.scatter_chart(
    x="floors",
    y='price',
    data = df,
    color=["#ffaa0088"]
    )


#------------------------------------- Feature Engineering  -----------------------------------------
    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# encoding the columns (waterfront, view , condition)

from sklearn.preprocessing import LabelEncoder

# encoding the condition column
le = LabelEncoder()
df['condition'] = le.fit_transform(df['condition'])

le = LabelEncoder()
df['view'] = le.fit_transform(df['view'])

le = LabelEncoder()
df['waterfront'] = le.fit_transform(df['waterfront'])


#-------------------------------------  Model Selection -----------------------------------------


# Prepare the data
X = df.drop(['price'], axis=1)
y = df['price']

#splitting the data into training and testing

from sklearn.model_selection import train_test_split
x_train,X_test,y_train,Y_test = train_test_split(X , y , test_size =0.2 ,random_state = 0 )

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting for results
y_pred = regressor.predict(X_test)

#calculating the R squared value
from sklearn.metrics import r2_score
#st.write(print("Accuracy:",r2_score(Y_test, y_pred)))


#----------------- feature engineering input ----------------------------


# User Input for Independent Variables
st.sidebar.header("Enter values to be Predicted", divider='rainbow')

# Create input boxes for each feature
user_input = {}
for feature in df.columns[:-1]:  # Exclude the target column
    user_input[feature] = st.sidebar.text_input(f"Enter {feature}")

# Button to trigger prediction
if st.sidebar.button("Predict"):
    # Create a DataFrame from user input
    user_input_df = pd.DataFrame([user_input], dtype=float)

    # Predict using the trained model
    y_pred = regressor.predict(user_input_df)


    # Predict using the trained model
    y_pred = regressor.predict(user_input_df)

    # Display the predicted values
    st.header("Predicted Price Fom Your Input:", divider='rainbow')
    st.write(y_pred[0])

 