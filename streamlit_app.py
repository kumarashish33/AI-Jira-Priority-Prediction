import streamlit as st

st.title("🤖 AI-Powered Jira Ticket Prioritization")
st.info("""
Example:

• Authentication failed when attempting Fetch command

• OAuth token keeps expiring

• pull push fail atlassian logon window open sourcetree push

• fake 3410 version sourcetree
""")

st.sidebar.title("About")

st.sidebar.write("""
AI Powered Jira Ticket Prioritization

Model:
- TF-IDF
- Logistic Regression

Developed using

- Python
- Scikit-learn
- Streamlit
""")

st.sidebar.metric(
    "Model Accuracy",
    "99.47%"
)

st.sidebar.metric(
    "Algorithm",
    "Logistic Regression"
)

st.write("Enter a Jira ticket summary below.")

ticket = st.text_area(
    "Jira Ticket Summary",
    height=150
)

predict_button = st.button("Predict Priority")

from src.predictor import predict_priority

if predict_button:

    if ticket.strip() == "":
        st.error("Please enter a Jira ticket summary.")

    else:

        with st.spinner("Predicting Priority..."):

            priority, confidence, class_probabilities = predict_priority(ticket)

        if priority == "Highest":
            st.error(f"🚨 Predicted Priority : {priority}")

        elif priority == "High":
            st.warning(f"⚠️ Predicted Priority : {priority}")

        elif priority == "Medium":
            st.info(f"ℹ️ Predicted Priority : {priority}")

        else:
            st.success(f"✅ Predicted Priority : {priority}")

        st.progress(float(confidence))

        st.write(f"**Confidence:** {confidence:.2%}")

        st.subheader("Prediction Probabilities")

        st.bar_chart(class_probabilities)