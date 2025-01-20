import streamlit as st

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            raise st.error("Cannot divide by zero!")
        return num1 / num2
    elif operation == "mod":
        if num2 == 0:
            raise st.error("Cannot divide by zero!")
        return num1 % num2


num1 = st.number_input(label="Number 1")
operation = st.selectbox("Operation", options=["add", "subtract", "multiply", "divide", "mod"])
num2 = st.number_input(label="Number 2")

st.write(f"Output: {calculator(num1, operation, num2)}")