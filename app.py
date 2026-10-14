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

def bisection_method(f_expr, a, b, tol, max_iter):
    """
    Bisection Method: Finds a root within an interval [a, b] where f(a) and f(b) have opposite signs.
    Stops when |b - a|/2 < tol or max_iter reached.
    """
    f_func = sp.lambdify(x, f_expr, "numpy")
    rows = []
    
    fa = float(f_func(a))
    fb = float(f_func(b))

    if fa * fb >= 0:
        return pd.DataFrame(), None, "f(a) and f(b) do not have opposite signs."

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = float(f_func(c))
        error = abs(b - a) / 2
        rows.append([i, a, b, c, fa, fb, fc, error])
        
        if error < tol:
            return pd.DataFrame(rows, columns=["Iteration", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error |b-a|/2"]), c, None

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return pd.DataFrame(rows, columns=["Iteration", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error |b-a|/2"]), (a + b) / 2, None

def false_position_method(f_expr, a, b, tol, max_iter):
    """
    False Position Method: Similar to bisection but uses a secant line to estimate the root.
    Stops when |x_next - x_prev| < tol or max_iter reached.
    """
    f_func = sp.lambdify(x, f_expr, "numpy")
    rows = []
    
    fa = float(f_func(a))
    fb = float(f_func(b))

    if fa * fb >= 0:
        return pd.DataFrame(), None, "f(a) and f(b) do not have opposite signs."

    x_prev = a 
    for i in range(1, max_iter + 1):
        if fb - fa == 0:
            return pd.DataFrame(), None, "Division by zero (f(b) - f(a) is zero)."

        c = b - fb * (b - a) / (fb - fa)
        fc = float(f_func(c))
        error = abs(c - x_prev)
        rows.append([i, a, b, c, fa, fb, fc, error])

        if error < tol:
            return pd.DataFrame(rows, columns=["Iteration", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error |c - x_prev|"]), c, None

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        x_prev = c
            
    return pd.DataFrame(rows, columns=["Iteration", "a", "b", "c", "f(a)", "f(b)", "f(c)", "Error |c - x_prev|"]), c, None

def secant_method(f_expr, x0, x1, tol, max_iter):
    """
    Secant Method: Approximates the derivative using two previous points.
    Stops when |x_next - x_n| < tol or max_iter reached.
    """
    f_func = sp.lambdify(x, f_expr, "numpy")
    rows = []

    for i in range(1, max_iter + 1):
        fx0 = float(f_func(x0))
        fx1 = float(f_func(x1))

        if fx1 - fx0 == 0:
            return pd.DataFrame(), None, "Division by zero (f(x1) - f(x0) is zero)."

        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x_next - x1)
        rows.append([i, x0, x1, fx0, fx1, x_next, error])

        if error < tol:
            return pd.DataFrame(rows, columns=["Iteration", "x_n-1", "x_n", "f(x_n-1)", "f(x_n)", "x_n+1", "Error |x_n+1 - x_n|"]), x_next, None

        x0 = x1
        x1 = x_next
            
    return pd.DataFrame(rows, columns=["Iteration", "x_n-1", "x_n", "f(x_n-1)", "f(x_n)", "x_n+1", "Error |x_n+1 - x_n|"]), x_next, None


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
    method = st.radio("Choose an algorithm", ["Newton-Raphson", "Fixed Point Iteration", "Bisection Method", "False Position Method", "Secant Method"])

    if method == "Newton-Raphson" or method == "Bisection Method" or method == "False Position Method":
        func_text = st.text_input("f(x) =", value="x**3 - x - 2", help="e.g. x**3 - x - 2, cos(x) - x, exp(x) - 3*x")
    elif method == "Fixed Point Iteration":
        func_text = st.text_input("g(x) =", value="(x + 2)**(1/3)", help="Rearrange f(x)=0 into x = g(x). e.g. (x+2)**(1/3)")
    else: # Secant Method
        func_text = st.text_input("f(x) =", value="x**3 - x - 2", help="e.g. x**3 - x - 2, cos(x) - x, exp(x) - 3*x")

    if method == "Bisection Method" or method == "False Position Method":
        a = st.number_input("Interval start (a)", value=1.0, format="%.6f")
        b = st.number_input("Interval end (b)", value=2.0, format="%.6f")
    elif method == "Secant Method":
        x0_secant = st.number_input("Initial guess x₀", value=1.0, format="%.6f")
        x1_secant = st.number_input("Second guess x₁", value=2.0, format="%.6f")
    else:
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

    elif method == "Newton-Raphson":
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
        
    elif method == "Bisection Method":
        df, root, error_message = bisection_method(expr, a, b, tol, max_iter)
        if error_message:
            st.error(error_message)
            st.stop()
        with col1:
            st.subheader("Iteration Table")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Result")
            last_err = df["Error |b-a|/2"].dropna().iloc[-1] if not df["Error |b-a|/2"].dropna().empty else None
            if last_err is not None and last_err < tol:
                st.success(f"✅ Converged to root ≈ **{root:.8f}** in {len(df)} iterations")
            else:
                st.warning(f"⚠️ Did not converge within {max_iter} iterations (last root approximation ≈ {root:.6f}).")
        
        # Plotting for Bisection Method
        f_func = sp.lambdify(x, expr, "numpy")
        fig = go.Figure()
        
        # Determine plot range
        min_x = min(df["a"].min(), df["b"].min(), root if root else a) - 1
        max_x = max(df["a"].max(), df["b"].max(), root if root else b) + 1
        grid = np.linspace(min_x, max_x, 400)
        
        try:
            fig.add_trace(go.Scatter(x=grid, y=f_func(grid), mode="lines", name="f(x)", line=dict(color="#2563eb")))
        except Exception:
            pass
        fig.add_trace(go.Scatter(x=grid, y=[0]*len(grid), mode="lines", name="y = 0", line=dict(color="gray", dash="dash")))

        fig.update_layout(title="Bisection Method Convergence", xaxis_title="x", yaxis_title="f(x)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif method == "False Position Method":
        df, root, error_message = false_position_method(expr, a, b, tol, max_iter)
        if error_message:
            st.error(error_message)
            st.stop()
        with col1:
            st.subheader("Iteration Table")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Result")
            last_err = df["Error |c - x_prev|"].dropna().iloc[-1] if not df["Error |c - x_prev|"].dropna().empty else None
            if last_err is not None and last_err < tol:
                st.success(f"✅ Converged to root ≈ **{root:.8f}** in {len(df)} iterations")
            else:
                st.warning(f"⚠️ Did not converge within {max_iter} iterations (last root approximation ≈ {root:.6f}).")

        # Plotting for False Position Method
        f_func = sp.lambdify(x, expr, "numpy")
        fig = go.Figure()
        
        min_x = min(df["a"].min(), df["b"].min(), root if root else a) - 1
        max_x = max(df["a"].max(), df["b"].max(), root if root else b) + 1
        grid = np.linspace(min_x, max_x, 400)
        
        try:
            fig.add_trace(go.Scatter(x=grid, y=f_func(grid), mode="lines", name="f(x)", line=dict(color="#2563eb")))
        except Exception:
            pass
        fig.add_trace(go.Scatter(x=grid, y=[0]*len(grid), mode="lines", name="y = 0", line=dict(color="gray", dash="dash")))

        for _, row in df.iterrows():
            if pd.isna(row.get("c")):
                continue
            a_, b_, fa_, fb_, c_ = row["a"], row["b"], row["f(a)"], row["f(b)"], row["c"]
            # Plot the secant line
            fig.add_trace(go.Scatter(x=[a_, b_], y=[fa_, fb_], mode="lines",
                                      line=dict(color="#dc2626", dash="dot"), showlegend=False))
            # Plot the new point on the x-axis
            fig.add_trace(go.Scatter(x=[c_], y=[0], mode="markers",
                                      marker=dict(color="#dc2626", size=8), showlegend=False))
        
        fig.update_layout(title="False Position Method Convergence", xaxis_title="x", yaxis_title="f(x)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif method == "Secant Method":
        df, root, error_message = secant_method(expr, x0_secant, x1_secant, tol, max_iter)
        if error_message:
            st.error(error_message)
            st.stop()
        with col1:
            st.subheader("Iteration Table")
            st.dataframe(df, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("Result")
            last_err = df["Error |x_n+1 - x_n|"].dropna().iloc[-1] if not df["Error |x_n+1 - x_n|"].dropna().empty else None
            if last_err is not None and last_err < tol:
                st.success(f"✅ Converged to root ≈ **{root:.8f}** in {len(df)} iterations")
            else:
                st.warning(f"⚠️ Did not converge within {max_iter} iterations (last root approximation ≈ {root:.6f}).")

        # Plotting for Secant Method
        f_func = sp.lambdify(x, expr, "numpy")
        fig = go.Figure()
        
        min_x = min(df["x_n-1"].min(), df["x_n"].min(), root if root else x0_secant) - 1
        max_x = max(df["x_n-1"].max(), df["x_n"].max(), root if root else x1_secant) + 1
        grid = np.linspace(min_x, max_x, 400)
        
        try:
            fig.add_trace(go.Scatter(x=grid, y=f_func(grid), mode="lines", name="f(x)", line=dict(color="#2563eb")))
        except Exception:
            pass
        fig.add_trace(go.Scatter(x=grid, y=[0]*len(grid), mode="lines", name="y = 0", line=dict(color="gray", dash="dash")))

        for _, row in df.iterrows():
            if pd.isna(row.get("x_n+1")):
                continue
            x0_curr, x1_curr, fx0_curr, fx1_curr, x_next_curr = row["x_n-1"], row["x_n"], row["f(x_n-1)"], row["f(x_n)"], row["x_n+1"]
            # Plot the secant line
            fig.add_trace(go.Scatter(x=[x0_curr, x1_curr], y=[fx0_curr, fx1_curr], mode="lines",
                                      line=dict(color="#dc2626", dash="dot"), showlegend=False))
            # Plot the new point on the x-axis
            fig.add_trace(go.Scatter(x=[x_next_curr], y=[0], mode="markers",
                                      marker=dict(color="#dc2626", size=8), showlegend=False))

        fig.update_layout(title="Secant Method Convergence", xaxis_title="x", yaxis_title="f(x)", height=450)
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
    | Solve x² - 3 = 0, in [1, 2] | `x**2 - 3` | N/A |
    | Solve x*exp(x) - 1 = 0, in [0, 1] | `x*exp(x) - 1` | N/A |
    """)

st.divider()
st.caption("Built for the Numerical Methods group project — implements Fixed Point Iteration, Newton-Raphson, Bisection Method, False Position Method & Secant Method from scratch in Python.")
