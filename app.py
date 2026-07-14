import streamlit as st
import time
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="IMDb Sentiment Analyzer", layout="wide")

st.title("🎬 IMDb Sentiment Analyzer")
st.markdown("Analyze movie reviews in real-time to determine if the sentiment is Positive or Negative.")

# New Feature: Recent History
if "history" not in st.session_state:
    st.session_state.history = []

col_input, col_stats = st.columns([2, 1])

with col_input:
    review = st.text_area("Enter your movie review here:", height=150)
    if st.button("Analyze Sentiment"):
        if not review:
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Analyzing sentiment..."):
                time.sleep(1)
                lower_review = review.lower()
                positive_words = ["good", "great", "best", "amazing", "love", "excellent", "fantastic"]
                negative_words = ["bad", "terrible", "worst", "awful", "hate", "boring", "waste"]
                
                pos_count = sum(1 for w in positive_words if w in lower_review)
                neg_count = sum(1 for w in negative_words if w in lower_review)
                
                if pos_count >= neg_count:
                    sentiment = "Positive"
                    confidence = 0.75 + (pos_count * 0.05)
                else:
                    sentiment = "Negative"
                    confidence = 0.70 + (neg_count * 0.05)
                    
                confidence = min(0.99, confidence)
                st.session_state.history.insert(0, {"review": review, "sentiment": sentiment, "confidence": confidence})
                
                st.success(f"**Result:** {sentiment} {'🌟' if sentiment == 'Positive' else '🍅'}")
                st.info(f"**Confidence:** {confidence*100:.1f}%")

with col_stats:
    if st.session_state.history:
        st.subheader("Sentiment Distribution")
        df_history = pd.DataFrame(st.session_state.history)
        fig = px.pie(df_history, names='sentiment', color='sentiment',
                     color_discrete_map={'Positive': '#4caf50', 'Negative': '#f44336'},
                     hole=0.4)
        fig.update_layout(showlegend=False, height=300, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

if st.session_state.history:
    st.divider()
    st.subheader("Recent Analysis History")
    for item in st.session_state.history[:5]:
        st.markdown(f"**{item['sentiment']}** (Conf: {item['confidence']*100:.1f}%) - _{item['review'][:100]}..._")
