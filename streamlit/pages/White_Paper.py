import streamlit as st
import sys

sys.path.append("../")
from utils.streamlit_helpers import read_markdown_file

white_paper_text = read_markdown_file("content/whitepaper.md")
st.markdown(white_paper_text)
