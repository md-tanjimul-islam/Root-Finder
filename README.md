# Root-Finding Visualizer

A Streamlit web app implementing five numerical root-finding algorithms:
- **Fixed Point Iteration**
- **Newton-Raphson Method**
- **Bisection Method**
- **False Position Method**
- **Secant Method**

Users can type **any function**, choose a method, and see the iteration table,
convergence graph, and final root.

## Features
- **Multiple Methods**: Choose from 5 different root-finding algorithms
- **Function Input**: Enter any mathematical function using Python/SymPy syntax
- **Interactive Tables**: See detailed iteration steps with error calculations
- **Visual Convergence**: Watch algorithms converge (or diverge) with intuitive plots
- **Automatic Differentiation**: For Newton-Raphson, derivatives are computed symbolically
- **Error Handling**: Graceful handling of invalid inputs and non-convergent cases
- **User Authentication**: Secure login system to track individual user history
- **User History**: Each user's activity is saved for future reference
- **Monetization Ready**: Integrated Google AdSense placeholder (replace with your own ad code)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (no credit card)
Quickest path: push this folder to a public GitHub repo, then deploy on
https://share.streamlit.io (Streamlit Community Cloud).

## Authentication
The app includes a simple login system using `streamlit-authenticator`. 
Default credentials (for testing):
- Username: admin
- Password: password

To change credentials, edit the `config.yaml` file and generate new password hashes using:
```python
import streamlit_authenticator as stauth
hasher = stauth.Hasher()
hasher.generate(['new_password1', 'new_password2'])
```

## User History
Each user's activity (function entered, method used, timestamp, and result) is saved to `user_history.json` in the app directory.
Note: On Streamlit Community Cloud, the file system is ephemeral; for persistent storage across app restarts, consider using a database.

## AdSense Integration
The app includes a placeholder for Google AdSense. To enable ads:
1. Sign up for Google AdSense and create an ad unit
2. Replace the placeholder ad code in `app.py` with your actual AdSense code
3. The ad will appear at the bottom of the app

## Methods Overview
1. **Newton-Raphson**: Uses derivative information for fast convergence (requires f'(x))
2. **Fixed Point Iteration**: Solves x = g(x) form (requires rearranging f(x)=0)
3. **Bisection Method**: Reliable bracketing method (requires sign change in interval)
4. **False Position Method**: Improves bisection with linear interpolation
5. **Secant Method**: Derivative-free approximation using two points

## Example Problems
| Goal | Newton-Raphson f(x) | Fixed Point g(x) | Bisection/False Position/Secant f(x) |
|------|-------------------|------------------|--------------------------------------|
| Solve x³ - x - 2 = 0 | `x**3 - x - 2` | `(x + 2)**(1/3)` | `x**3 - x - 2` |
| Solve cos(x) - x = 0 | `cos(x) - x` | `cos(x)` | `cos(x) - x` |
| Solve e^x - 3x = 0 | `exp(x) - 3*x` | `exp(x)/3` | `exp(x) - 3*x` |
| Solve x² - 3 = 0, in [1, 2] | `x**2 - 3` | N/A | `x**2 - 3` |
| Solve x*exp(x) - 1 = 0, in [0, 1] | `x*exp(x) - 1` | N/A | `x*exp(x) - 1` |

## Built With
- [Streamlit](https://streamlit.io) - For the web interface
- [SymPy](https://www.sympy.org) - Symbolic mathematics library
- [NumPy](https://numpy.org) - Numerical computations
- [Pandas](https://pandas.pydata.org) - Data tables
- [Plotly](https://plotly.com) - Interactive plots
- [streamlit-authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) - Authentication

---
*Built for numerical methods education - visualize root-finding algorithms in action!*