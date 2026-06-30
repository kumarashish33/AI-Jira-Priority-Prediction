import streamlit as st

st.markdown("""
<style>

/* Background */
.stApp{
    background-color:#0E1117;
}

/* Button */
.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D4ED8;
}

/* Text Area */
.stTextArea textarea {
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#1A1D29;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🤖 AI-Powered Jira Ticket Prioritization
### Intelligent NLP-based Priority Prediction for Jira Issues
""")
st.info("""
💡 Example Tickets

• Authentication failed when attempting Fetch command

• OAuth token keeps expiring

• Pull push failed after Atlassian login

• Fake 3410 version Sourcetree

""")

st.sidebar.title("🤖 AI Jira Prioritization")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model")
st.sidebar.write("• TF-IDF")
st.sidebar.write("• Logistic Regression")

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Tech Stack")
st.sidebar.write("• Python")
st.sidebar.write("• Scikit-learn")
st.sidebar.write("• Streamlit")

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Performance")
st.sidebar.write("Accuracy: **99.47%**")

st.sidebar.markdown("---")

st.sidebar.subheader("🚀 Version")
st.sidebar.success("Version 1.0")

st.write("Enter a Jira ticket summary below.")

ticket = st.text_area(
    "Jira Ticket Summary",
    height=150
)

predict_button = st.button(
    "🚀 Predict Priority",
    use_container_width=True
)



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

        st.markdown(f"### 📈 Confidence Score: **{confidence:.2%}**")

        st.markdown("## 📊 Prediction Probabilities")

        st.bar_chart(class_probabilities)