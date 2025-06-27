import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import comb

# =========================================================
# ================ CST CORE AND UTILITIES ================
# =========================================================

def class_function(x, N1=0.5, N2=1.0):
    return (x ** N1) * ((1 - x) ** N2)

def bernstein_polynomial(n, i, x):
    return comb(n, i) * (x ** i) * (1 - x) ** (n - i)

def shape_function(x, A):
    n = len(A) - 1
    return sum(A[i] * bernstein_polynomial(n, i, x) for i in range(n + 1))

def cst_function(x, A, N1=0.5, N2=1.0):
    return class_function(x, N1, N2) * shape_function(x, A)

def cosine_spacing(n_points=200):
    beta = np.linspace(0, np.pi, n_points)
    return 0.5 * (1 - np.cos(beta))

def safe_normalize(x):
    dx = x[-1] - x[0]
    if abs(dx) < 1e-6 or len(x) < 2:
        return None
    return (x - x[0]) / dx

def fit_cst(x, y, order=6):
    def residuals(A):
        return cst_function(x, A) - y
    A0 = np.zeros(order + 1)
    result = least_squares(residuals, A0)
    return result.x

def load_airfoil(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    coords = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
            except ValueError:
                continue
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]

def split_upper_lower(x, y):
    idx_le = np.argmin(x)
    return (x[:idx_le+1], y[:idx_le+1]), (x[idx_le:], y[idx_le:])

# =========================================================
# ================ FITTING FUNCTION =======================
# =========================================================

def analyze_airfoil_folder(folder_name="datFolder", order=6, plot=True):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    folder_path = os.path.join(script_dir, folder_name)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"'{folder_path}' does not exist. Please create it and add .dat files.")

    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".dat"):
            path = os.path.join(folder_path, filename)
            try:
                x, y = load_airfoil(path)
                (x_u, y_u), (x_l, y_l) = split_upper_lower(x, y)

                x_u_norm = safe_normalize(x_u[::-1])
                x_l_norm = safe_normalize(x_l)

                if x_u_norm is None or x_l_norm is None:
                    print(f"Skipping {filename}: invalid or degenerate x-coordinates")
                    continue

                A_upper = fit_cst(x_u_norm, y_u[::-1], order)
                A_lower = fit_cst(x_l_norm, y_l, order)

                results.append({
                    "airfoil": filename,
                    "A_upper": A_upper,
                    "A_lower": A_lower
                })

                if plot:
                    output_plot_dir = os.path.join(script_dir, "NACA_CST_fittedCoeff_pic")
                    os.makedirs(output_plot_dir, exist_ok=True)

                    x_dense = np.linspace(0, 1, 200)
                    y_fit_u = cst_function(x_dense, A_upper)
                    y_fit_l = cst_function(x_dense, A_lower)

                    plt.figure(figsize=(10, 4))
                    plt.plot(x_u_norm, y_u[::-1], 'b.', label='Upper Original')
                    plt.plot(x_l_norm, y_l, 'g.', label='Lower Original')
                    plt.plot(x_dense, y_fit_u, 'b-', label='Upper CST Fit')
                    plt.plot(x_dense, y_fit_l, 'g-', label='Lower CST Fit')
                    plt.axis('equal')
                    plt.grid(True)
                    plt.title(f"CST Fit: {filename}")
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.legend()
                    plt.tight_layout()

                    image_path = os.path.join(output_plot_dir, f"{os.path.splitext(filename)[0]}.png")
                    plt.savefig(image_path, dpi=300)
                    plt.close()

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                continue

    # Save all coefficients to CSV
    data = []
    for r in results:
        row = [r["airfoil"]]
        row.extend(r["A_upper"])
        row.extend(r["A_lower"])
        data.append(row)

    cols = ["airfoil"] + [f"A_u{i}" for i in range(order + 1)] + [f"A_l{i}" for i in range(order + 1)]
    df = pd.DataFrame(data, columns=cols)
    output_path = os.path.join(script_dir, "cst_fitted_coefficients.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved fitted CST coefficients to: {output_path}")

# =========================================================
# ================ CHECKING FUNCTION ======================
# =========================================================

def compute_geometry(A_upper, A_lower, n_points=200):
    x = cosine_spacing(n_points)
    y_u = cst_function(x, A_upper)
    y_l = cst_function(x, A_lower)
    thickness = y_u - y_l
    camber = (y_u + y_l) / 2
    return x, y_u, y_l, thickness, camber

def plot_airfoil_with_artifacts(airfoil_name, x, y_u, y_l, thickness):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    output_plot_dir = os.path.join(script_dir, "CST_coefficientCheck_pic")
    os.makedirs(output_plot_dir, exist_ok=True)

    plt.figure(figsize=(10,5))
    plt.plot(x, y_u, label="Upper Surface")
    plt.plot(x, y_l, label="Lower Surface")
    plt.fill_between(x, y_l, y_u, color='lightgray', alpha=0.5)
    plt.plot(x[thickness <= 0], y_u[thickness <= 0], 'ro', label="Thickness ≤ 0", markersize=6)
    plt.plot(x[thickness <= 0], y_l[thickness <= 0], 'ro', markersize=6)
    plt.title(f"Airfoil: {airfoil_name}\nPoints with Zero or Negative Thickness Highlighted")
    plt.xlabel("Chordwise Location (x)")
    plt.ylabel("Y Coordinate")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    image_path = os.path.join(output_plot_dir, f"{os.path.splitext(airfoil_name)[0]}_check.png")
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()

def check_coefficients(order=6):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    csv_path = os.path.join(script_dir, "cst_fitted_coefficients.csv")
    df = pd.read_csv(csv_path)

    df[f"A_u{order}"] = 0.0
    df[f"A_l{order}"] = 0.0

    results = []
    for idx, row in df.iterrows():
        A_upper = row[[f"A_u{i}" for i in range(order+1)]].values
        A_lower = row[[f"A_l{i}" for i in range(order+1)]].values
        x, y_u, y_l, thickness, camber = compute_geometry(A_upper, A_lower)
        bad_points_count = np.sum(thickness[1:-1] <= 0)
        results.append({"airfoil": row["airfoil"], "bad_points": bad_points_count})

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("airfoil_thickness_issues_summary.csv", index=False)
    print("\nSummary CSV saved as 'airfoil_thickness_issues_summary.csv'.")

    results_sorted = sorted(results, key=lambda r: r["bad_points"], reverse=True)
    print("Top 10 airfoils with the most zero or negative thickness points:")
    for i in range(min(10, len(results_sorted))):
        r = results_sorted[i]
        print(f"{i+1}. {r['airfoil']} - {r['bad_points']} bad points")
        row = df[df["airfoil"] == r["airfoil"]].iloc[0]
        A_upper = row[[f"A_u{i}" for i in range(order+1)]].values
        A_lower = row[[f"A_l{i}" for i in range(order+1)]].values
        x, y_u, y_l, thickness, camber = compute_geometry(A_upper, A_lower)
        plot_airfoil_with_artifacts(r["airfoil"], x, y_u, y_l, thickness)

# =========================================================
# ================= MAIN EXECUTION ========================
# =========================================================

if __name__ == "__main__":
    analyze_airfoil_folder("datFolder", order=6, plot=True)
    check_coefficients(order=6)
