"""
Root-Finding Visualizer
Fixed Point Iteration & Newton-Raphson Method
------------------------------------------------
A Streamlit web app that lets a user type in ANY function, pick a method,
and watch the algorithm converge (or diverge) to a root — complete with
an iteration table and a convergence graph.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fixed Point & Newton-Raphson Visualizer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Root-Finding Algorithm Visualizer")
st.caption("Fixed Point Iteration  &  Newton-Raphson Method — type any function and watch it converge.")

x = sp.Symbol("x")


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def parse_expr(text):
    """Safely parse a user-typed math expression into a sympy expression."""
    transformations = sp.parsing.sympy_parser.standard_transformations + (
        sp.parsing.sympy_parser.implicit_multiplication_application,
        sp.parsing.sympy_parser.convert_xor,
    )
    expr = sp.parsing.sympy_parser.parse_expr(text, transformations=transformations)
    return expr


def fixed_point_iteration(g_expr, x0, tol, max_iter):
    """
    Fixed Point Iteration:  x_{n+1} = g(x_n)
    Stops when |x_{n+1} - x_n| < tol  or  max_iter reached.
    """
    g_func = sp.lambdify(x, g_expr, "numpy")
    rows = []
    xn = x0
    for i in range(1, max_iter + 1):
        try:
            x_next = float(g_func(xn))
        except Exception:
            rows.append([i, xn, None, None])
            break
        error = abs(x_next - xn)
        rows.append([i, xn, x_next, error])
        if error < tol or not np.isfinite(x_next):
            xn = x_next
            break
        xn = x_next
    df = pd.DataFrame(rows, columns=["Iteration", "x_n", "x_(n+1) = g(x_n)", "Error |x_(n+1) - x_n|"])
    return df, xn


def newton_raphson(f_expr, x0, tol, max_iter):
    """
    Newton-Raphson:  x_{n+1} = x_n - f(x_n)/f'(x_n)
    Stops when |x_{n+1} - x_n| < tol  or  max_iter reached.
    """
    f_prime_expr = sp.diff(f_expr, x)
    f_func = sp.lambdify(x, f_expr, "numpy")
    fp_func = sp.lambdify(x, f_prime_expr, "numpy")

    rows = []
    xn = x0
    for i in range(1, max_iter + 1):
        try:
            fxn = float(f_func(xn))
            fpxn = float(fp_func(xn))
            if fpxn == 0:
                rows.append([i, xn, fxn, fpxn, None, None])
                break
            x_next = xn - fxn / fpxn
        except Exception:
            rows.append([i, xn, None, None, None, None])
            break
        error = abs(x_next - xn)
        rows.append([i, xn, fxn, fpxn, x_next, error])
        if error < tol or not np.isfinite(x_next):
            xn = x_next
            break
        xn = x_next
    df = pd.DataFrame(rows, columns=["Iteration", "x_n", "f(x_n)", "f'(x_n)", "x_(n+1)", "Error"])
    return df, xn, f_prime_expr


# --------------------------------------------------------------------------
# Sidebar — user inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio("Choose an algorithm", ["Newton-Raphson", "Fixed Point Iteration"])

    if method == "Newton-Raphson":
        func_text = st.text_input("f(x) =", value="x**3 - x - 2", help="e.g. x**3 - x - 2, cos(x) - x, exp(x) - 3*x")
    else:
        func_text = st.text_input("g(x) =", value="(x + 2)**(1/3)", help="Rearrange f(x)=0 into x = g(x). e.g. (x+2)**(1/3)")

    x0 = st.number_input("Initial guess x₀", value=1.5, format="%.6f")
    tol = st.number_input("Tolerance (stopping error)", value=1e-6, format="%.8f")
    max_iter = st.slider("Maximum iterations", min_value=1, max_value=200, value=50)
    run = st.button("🚀 Run Algorithm", use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
if run:
    try:
        expr = parse_expr(func_text)
    except Exception as e:
        st.error(f"Could not parse your function: {e}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    if method == "Fixed Point Iteration":
        df, root = fixed_point_iteration(expr, x0, tol, max_iter)
        with col1:
            st.subheader("Iteration Table")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Result")
            last_err = df["Error |x_(n+1) - x_n|"].dropna().iloc[-1] if not df["Error |x_(n+1) - x_n|"].dropna().empty else None
            if last_err is not None and last_err < tol:
                st.success(f"✅ Converged to root ≈ **{root:.8f}** in {len(df)} iterations")
            else:
                st.warning(f"⚠️ Did not converge within {max_iter} iterations (last x ≈ {root:.6f}). "
                           f"Try a different g(x) or initial guess — fixed point iteration only converges if |g'(x)| < 1 near the root.")

        # Cobweb-style convergence plot
        xs = df["x_n"].tolist()
        fig = go.Figure()
        g_func = sp.lambdify(x, expr, "numpy")
        lo, hi = min(xs + [root]) - 2, max(xs + [root]) + 2
        grid = np.linspace(lo, hi, 400)
        try:
            fig.add_trace(go.Scatter(x=grid, y=g_func(grid), mode="lines", name="g(x)", line=dict(color="#2563eb")))
        except Exception:
            pass
        fig.add_trace(go.Scatter(x=grid, y=grid, mode="lines", name="y = x", line=dict(color="gray", dash="dash")))

        cobweb_x, cobweb_y = [], []
        cur = x0
        for val in df["x_(n+1) = g(x_n)"].tolist():
            if val is None:
                break
            cobweb_x += [cur, cur, val]
            cobweb_y += [g_func(cur), val, val]
            cur = val
        fig.add_trace(go.Scatter(x=cobweb_x, y=cobweb_y, mode="lines+markers", name="Iteration path", line=dict(color="#dc2626")))
        fig.update_layout(title="Cobweb Diagram — Fixed Point Iteration", xaxis_title="x", yaxis_title="g(x)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Newton-Raphson
        df, root, f_prime_expr = newton_raphson(expr, x0, tol, max_iter)
        st.info(f"f'(x) = {sp.sstr(f_prime_expr)}  (computed automatically via symbolic differentiation)")
        with col1:
            st.subheader("Iteration Table")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Result")
            last_err = df["Error"].dropna().iloc[-1] if not df["Error"].dropna().empty else None
            if last_err is not None and last_err < tol:
                st.success(f"✅ Converged to root ≈ **{root:.8f}** in {len(df)} iterations")
            else:
                st.warning(f"⚠️ Did not converge within {max_iter} iterations (last x ≈ {root:.6f}). "
                           f"Try a different initial guess — Newton-Raphson can fail if f'(x₀) is 0 or the guess is far from the root.")

        # Tangent-line convergence plot
        f_func = sp.lambdify(x, expr, "numpy")
        xs = df["x_n"].dropna().tolist()
        fig = go.Figure()
        lo, hi = min(xs + [root]) - 2, max(xs + [root]) + 2
        grid = np.linspace(lo, hi, 400)
        try:
            fig.add_trace(go.Scatter(x=grid, y=f_func(grid), mode="lines", name="f(x)", line=dict(color="#2563eb")))
        except Exception:
            pass
        fig.add_trace(go.Scatter(x=grid, y=[0]*len(grid), mode="lines", name="y = 0", line=dict(color="gray", dash="dash")))

        for _, row in df.iterrows():
            if pd.isna(row.get("x_(n+1)")):
                continue
            xn_, fxn_, x_next_ = row["x_n"], row["f(x_n)"], row["x_(n+1)"]
            fig.add_trace(go.Scatter(x=[xn_, x_next_], y=[fxn_, 0], mode="lines+markers",
                                      line=dict(color="#dc2626"), showlegend=False))
        fig.update_layout(title="Tangent Lines — Newton-Raphson Convergence", xaxis_title="x", yaxis_title="f(x)", height=450)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Enter a function and settings in the sidebar, then click **Run Algorithm**.")
    st.markdown("""
    ### How to use
    - **Newton-Raphson**: enter `f(x)` such that you want to solve `f(x) = 0`. The derivative is computed automatically.
    - **Fixed Point Iteration**: rearrange `f(x) = 0` into the form `x = g(x)` and enter `g(x)`.
    - Use Python-style math syntax: `x**2`, `sin(x)`, `cos(x)`, `exp(x)`, `sqrt(x)`, `log(x)`.

    ### Example problems
    | Goal | Newton-Raphson f(x) | Fixed Point g(x) |
    |---|---|---|
    | Solve x³ - x - 2 = 0 | `x**3 - x - 2` | `(x + 2)**(1/3)` |
    | Solve cos(x) - x = 0 | `cos(x) - x` | `cos(x)` |
    | Solve e^x - 3x = 0 | `exp(x) - 3*x` | `exp(x)/3` |
    """)

st.divider()
st.caption("Built for the Numerical Methods group project — implements Fixed Point Iteration & Newton-Raphson from scratch in Python.")
