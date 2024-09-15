import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title("Pressure Vector Calculator")

# Short description using an expander for instructions
with st.expander("How it works"):
    st.write("""
    This app allows you to:
    
    1. Upload a data spreadsheet.
    2. Input a value for W.
    3. Generate various matrices (A, B, C, Z) using provided formulas.
    4. Calculate the pressure vector and plot the arctangent of the pressure terms.
    
    Once you upload the file and input the value for W, the app will automatically generate the necessary matrices and the chart.
    """)

# Step 1: User input for W
W = st.number_input("Enter the value for W", min_value=1, value=1000, step=1)

# Step 2: File uploader for the data
uploaded_file = st.file_uploader("Upload a data spreadsheet in .xlsx format", type=["xlsx"])

# Placeholder for data processing and chart generation
if uploaded_file is not None:
    # Step 3: Read the uploaded spreadsheet
    data = pd.read_excel(uploaded_file, sheet_name='Sheet1')

    # Extract the necessary columns from the uploaded file based on the correct column names
    m_values = data['m'].astype(float).values
    n_values = data['n'].astype(float).values
    p_values = data['p'].astype(float).values
    q_values = data['q'].astype(float).values
    V_values = data['V'].astype(float).values
    velocity_vector = data['vel'].astype(float).values

    # Calculate matrices A, B, C, Z, pressure vector, and arctangent values
    matrix_A = []
    for p, q in zip(p_values, q_values):
        column = []
        for m, n in zip(m_values, n_values):
            term = (2 * np.pi / W) * (p * m + q * n)
            g_mnpq = np.cos(term) - 1j * np.sin(term)
            column.append(g_mnpq)
        matrix_A.append(column)
    matrix_A = np.array(matrix_A).T

    # Matrix C is the Hermitian of A
    matrix_C = np.conjugate(matrix_A).T

    # Matrix B is the diagonal matrix of V values
    matrix_B = np.diag(V_values)

    # Calculate matrix Z
    matrix_Z = np.dot(np.dot(matrix_A, matrix_B), matrix_C)

    # Compute the pressure vector
    pressure_vector = np.dot(matrix_Z, velocity_vector)

    # Calculate the arctangent of the pressure vector
    arctangent_values = np.arctan2(pressure_vector.imag, pressure_vector.real)

    # Generate the chart
    indexing_numbers = np.arange(1, len(pressure_vector) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(indexing_numbers, arctangent_values, marker='o', linestyle='-', color='b')
    plt.title('Arctangent of Pressure Terms as a Function of Indexing Number')
    plt.xlabel('Indexing Number')
    plt.ylabel('Arctangent (Imaginary P / Real P)')
    plt.grid(True)

    # Display the chart
    st.pyplot(plt)

    # Display Matrices A, B, C, and Pressure Vector P
    st.write("### Matrix A")
    st.write(pd.DataFrame(matrix_A))

    st.write("### Matrix B")
    st.write(pd.DataFrame(matrix_B))

    st.write("### Matrix C (Hermitian of A)")
    st.write(pd.DataFrame(matrix_C))

    # Display Matrix Z
    st.write("### Matrix Z (A x B x C)")
    st.write(pd.DataFrame(matrix_Z))

    st.write("### Pressure Vector P (Real and Imaginary parts)")
    df_pressure_vector = pd.DataFrame({
        "Real Part of P": pressure_vector.real,
        "Imaginary Part of P": pressure_vector.imag
    })
    st.write(df_pressure_vector)
