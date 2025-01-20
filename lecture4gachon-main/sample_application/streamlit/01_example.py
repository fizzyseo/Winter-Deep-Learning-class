import streamlit as st

name = st.text_input("Name")
is_morning = st.checkbox("Is morning", False)

if name:
    salutation = "Good morning" if is_morning else "Good evening"
    st.write(f"{salutation}, {name}.")