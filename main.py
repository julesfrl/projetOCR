import streamlit as st
from page1 import page1
from page2 import page2
from page3 import page3
from page4 import page4

st.title("Projet OCR - OCT 20 DS")
st.subheader(" ")
page_selectionnee = st.sidebar.selectbox(
    label="Page", options=["Introduction","Galerie","Démonstration", "Amélioration"]
)

if page_selectionnee == "Introduction":
    page1()
elif page_selectionnee == "Galerie":
    page2()
elif page_selectionnee == "Démonstration":
    page3()
else:
    page4()
 
