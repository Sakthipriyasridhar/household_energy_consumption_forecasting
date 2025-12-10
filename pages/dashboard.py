import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="🏠")
st.title("🏠 Dashboard")
st.markdown("### Your Energy Management Hub")

# This page is redundant since main.py handles dashboard
# Redirect to main
st.info("You're already on the dashboard! Use the sidebar to navigate.")
