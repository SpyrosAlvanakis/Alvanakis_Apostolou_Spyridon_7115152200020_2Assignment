import streamlit as st

# Custom imports for your pages
import str_hello
import str_expl
import str_clf_comp

# Initialize session state for page navigation if not already set
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Introduction'

# Set page configuration
st.set_page_config(page_title="Hepatitis C Data Exploration", page_icon="🧪")

# Sidebar navigation
st.sidebar.title("Navigation")
# Define your pages
pages = {
    "Introduction": str_hello,
    "Data Exploration": str_expl,
    "Estimators Comparison": str_clf_comp
}

# Option for users to select a page
page_option = st.sidebar.radio("Select a Page", list(pages.keys()))

# Update session state
st.session_state['current_page'] = page_option

# Load the selected page module
pages[st.session_state['current_page']].app()


