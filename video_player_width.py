import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 

def main():
    st.title("Player Width Quadratic and Linear Fits")
    
    reference_points = [ 
        {"width": 960, "factor": 0.857}, 
        {"width": 1024, "factor": 0.865}, 
        {"width": 1280, "factor": 0.893}, 
        {"width": 1320, "factor": 0.899}, 
        {"width": 1536, "factor": 0.911} 
    ] 
    
    st.markdown("### 📋 Data Points")
    st.write("Input data points used for the fitting process:")
    st.table(reference_points) 
 
    widths = np.array([point["width"] for point in reference_points]) 
    factors = np.array([point["factor"] for point in reference_points]) 
    
    def quadratic_model(x, a, b, c): 
        return a * x**2 + b * x + c 
    
    def linear_model(x, m, c): 
        return m * x + c 
    
    params_quadratic, cov_quadratic = curve_fit(quadratic_model, widths, factors) 
    params_linear, cov_linear = curve_fit(linear_model, widths, factors) 
    
    quadratic_errors = np.sqrt(np.diag(cov_quadratic))
    linear_errors = np.sqrt(np.diag(cov_linear))
    
    x_vals = np.linspace(min(widths), max(widths), 500) 
    y_quadratic = quadratic_model(x_vals, *params_quadratic) 
    y_linear = linear_model(x_vals, *params_linear) 
    

    st.markdown("### 📈 Visualization")
    st.write("Graph showing quadratic and linear model fits:")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(widths, factors, color='red', label="Data Points", s=100, zorder=5) 
    ax.plot(x_vals, y_quadratic, linestyle="--", label="Quadratic Fit", color="blue", linewidth=2) 
    ax.plot(x_vals, y_linear, linestyle=":", label="Linear Fit", color="green", linewidth=2) 
    ax.set_xlabel("Width", fontsize=12) 
    ax.set_ylabel("Factor", fontsize=12) 
    ax.set_title("Width vs Factor: Model Fits", fontsize=14)
    ax.legend(fontsize=10) 
    ax.grid(color='gray', linestyle='--', linewidth=0.5) 
    st.pyplot(fig) 
    
    a, b, c = params_quadratic 
    m, c_linear = params_linear 
    
    st.markdown("### 🧮 Interactive Prediction")
    st.write("Test the models by inputting a width:")

    input_width = st.number_input(
        "Enter Width:", 
        min_value=min(widths)-200, 
        max_value=max(widths)+200, 
        value=1200, 
        step=50
    )

    quadratic_factor = quadratic_model(input_width, a, b, c)
    linear_factor = linear_model(input_width, m, c_linear)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Quadratic Model Prediction:**")
        st.metric(
            label="Factor", 
            value=f"{quadratic_factor:.4f}"
        )
    
    with col2:
        st.markdown("**Linear Model Prediction:**")
        st.metric(
            label="Factor", 
            value=f"{linear_factor:.4f}"
        )
    
    st.markdown("### 📋 Detailed Model Information")
    

    st.markdown("**Quadratic Model:**")
    st.latex(r"\text{factor} = a \cdot \text{width}^2 + b \cdot \text{width} + c")
    st.write(f"a = {a:.6e} ± {quadratic_errors[0]:.6e}")
    st.write(f"b = {b:.6f} ± {quadratic_errors[1]:.6f}")  # Use normal float format
    st.write(f"c = {c:.6f} ± {quadratic_errors[2]:.6f}")
    

    st.markdown("**Linear Model:**")
    st.latex(r"\text{factor} = m \cdot \text{width} + c")
    st.write(f"m = {m:.6e} ± {linear_errors[0]:.6e}")
    st.write(f"c = {c_linear:.6f} ± {linear_errors[1]:.6f}")

if __name__ == "__main__":
    main()
