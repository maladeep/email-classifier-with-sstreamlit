# import streamlit as st

# # BMI calculation function
# def calculate_bmi(weight, height):
#     if height != 0:
#         height_m = height / 100  # Convert height from cm to meters
#         bmi = weight / (height_m ** 2)
#         return bmi
#     else:
#         return 0

# # Streamlit app
# def main():
#     st.title("BMI Calculator")

#     # User input
#     weight = st.number_input("Enter your weight (in kg)")
#     height = st.number_input("Enter your height (in cm)")

#     # Calculate BMI
#     bmi = calculate_bmi(weight, height)

#     # Display result
#     st.subheader("BMI Result")
#     st.write("Weight:", weight, "kg")
#     st.write("Height:", height, "cm")
    
#     if bmi != 0:
#         st.write("BMI:", bmi)
#     else:
#         st.write("Please enter a non-zero height value.")

#     # BMI Categories
#     st.subheader("BMI Categories")
#     st.write("Underweight: BMI < 18.5")
#     st.write("Normal weight: 18.5 <= BMI < 24.9")
#     st.write("Overweight: 25 <= BMI < 29.9")
#     st.write("Obesity: BMI >= 30")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB

# # Function to train the Naive Bayes model
# def train_model(data, labels):
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(data)
#     model = MultinomialNB()
#     model.fit(X, labels)
#     return model, vectorizer

# # Function to predict spam or ham
# def predict(model, vectorizer, message):
#     X = vectorizer.transform([message])
#     prediction = model.predict(X)
#     return prediction[0]

# # Streamlit app
# def main():
#     st.title("Spam or Ham Detection")

#     # Train the model
#     data = [
#         "Hey, how are you?",
#         "Free money! Click here now!",
#         "I'm going to the park.",
#         "Congratulations! You've won a prize!",
#         "Reminder: Meeting tomorrow at 2 PM."
#     ]
#     labels = ["ham", "spam", "ham", "spam", "ham"]
#     model, vectorizer = train_model(data, labels)

#     # User input
#     message = st.text_input("Enter a message")

#     # Predict
#     if st.button("Predict"):
#         prediction = predict(model, vectorizer, message)
#         st.write("Prediction:", prediction)

# if __name__ == "__main__":
#     main()




import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to train the Naive Bayes model and save it as a pickle file
def train_model(data, labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    model = MultinomialNB()
    model.fit(X, labels)

    # Save the model as a pickle file
    with open('spam_ham_model.pkl', 'wb') as file:
        pickle.dump((model, vectorizer), file)

    return model, vectorizer

# Function to load the trained model from the pickle file
def load_model():
    with open('spam_ham_model.pkl', 'rb') as file:
        model, vectorizer = pickle.load(file)
    return model, vectorizer

# Function to predict spam or ham
def predict(model, vectorizer, message):
    X = vectorizer.transform([message])
    prediction = model.predict(X)
    return prediction[0]

# Streamlit app
def main():
    st.title("Spam or Ham Detection")
    st.balloons() 

    # Check if the model pickle file exists
    if not os.path.exists('spam_ham_model.pkl'):
        # Train the model and create the pickle file
        data = [
            "Hey, how are you?",
            "Free money! Click here now!",
            "I'm going to the park.",
            "Congratulations! You've won a prize!",
            "Reminder: Meeting tomorrow at 2 PM."
        ]
        labels = ["ham", "spam", "ham", "spam", "ham"]
        model, vectorizer = train_model(data, labels)
    else:
        # Load the model from the pickle file
        model, vectorizer = load_model()

    # User input
    message = st.text_input("Enter a message")

    # Predict
    if st.button("Predict"):
        prediction = predict(model, vectorizer, message)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()



