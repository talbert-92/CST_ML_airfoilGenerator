import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
import random

# --- CST core functions ---
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

def compute_geometry_features(A_upper, A_lower, n_points=200):
    x = cosine_spacing(n_points)
    y_u = cst_function(x, A_upper)
    y_l = cst_function(x, A_lower)
    thickness = y_u - y_l
    camber_line = (y_u + y_l) / 2
    max_thickness = np.max(thickness)
    max_thickness_pos = x[np.argmax(thickness)]
    max_camber = np.max(np.abs(camber_line))
    max_camber_pos = x[np.argmax(np.abs(camber_line))]
    te_gap = y_u[-1] - y_l[-1]
    
    # Estimate LE radius by quadratic fit (can improve later if needed)
    points_for_fit = 5
    x_fit = x[:points_for_fit]
    y_fit = y_u[:points_for_fit]
    p = np.polyfit(x_fit, y_fit, 2)
    a = p[0]
    le_radius = 1 / (2 * a) if a != 0 else 0
    
    # Smoothness: check oscillations in dy/dx near LE
    dy_dx = np.gradient(y_u, x)
    smoothness_valid = np.all(np.abs(np.diff(dy_dx[:10])) < 0.2)
    
    return {
        "thickness": thickness,
        "max_thickness": max_thickness,
        "max_thickness_pos": max_thickness_pos,
        "max_camber": max_camber,
        "max_camber_pos": max_camber_pos,
        "te_gap": te_gap,
        "le_radius": le_radius,
        "smoothness_valid": smoothness_valid
    }

def compute_geometry(A_upper, A_lower, n_points=200):
    x = cosine_spacing(n_points)
    y_u = cst_function(x, A_upper)
    y_l = cst_function(x, A_lower)
    thickness = y_u - y_l
    camber = (y_u + y_l) / 2
    return x, y_u, y_l, thickness, camber

def plot_airfoil(name, x, y_u, y_l, thickness):
    plt.figure(figsize=(10,5))
    plt.plot(x, y_u, label="Upper Surface")
    plt.plot(x, y_l, label="Lower Surface")
    plt.fill_between(x, y_l, y_u, alpha=0.3)
    plt.plot(x[thickness <= 0], y_u[thickness <= 0], 'ro', label="Thickness ≤ 0")
    plt.plot(x[thickness <= 0], y_l[thickness <= 0], 'ro')
    plt.title(f"Invalid Airfoil Sample: {name}")
    plt.xlabel("Chordwise Location (x)")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def visualize_random_invalid_airfoils(csv_path, n_samples=5, order=6):
    df_invalid = pd.read_csv(csv_path)
    samples = random.sample(range(len(df_invalid)), min(n_samples, len(df_invalid)))
    
    for idx in samples:
        row = df_invalid.iloc[idx]
        A_upper = row[[f"A_u{i}" for i in range(order+1)]].values
        A_lower = row[[f"A_l{i}" for i in range(order+1)]].values
        
        x, y_u, y_l, thickness, camber = compute_geometry(A_upper, A_lower)
        plot_airfoil(f"Sample {idx}", x, y_u, y_l, thickness)


# --- Validity check function ---
def is_valid_airfoil(A_upper, A_lower):
    geom = compute_geometry_features(A_upper, A_lower)
    positive_thickness = np.all(geom["thickness"][1:-1] > 0)  # exclude LE & TE
    max_thickness_valid = 0.06 <= geom["max_thickness"] <= 0.15
    max_camber_valid = 0.0 <= geom["max_camber"] <= 0.06
    te_gap_valid = abs(geom["te_gap"]) < 1e-4
    le_radius_valid = 0.005 <= geom["le_radius"] <= 0.03
    smoothness_valid = geom["smoothness_valid"]
    return all([positive_thickness, max_thickness_valid, max_camber_valid, te_gap_valid, le_radius_valid, smoothness_valid])

# --- Generate invalid samples ---
def generate_invalid_samples(n_samples=600, order=6, bound=0.5, max_trials=10000):
    invalid_samples = []
    trials = 0

    while len(invalid_samples) < n_samples and trials < max_trials:
        trials += 1
        A_upper = np.random.uniform(-bound, bound, order)
        A_upper = np.append(A_upper, 0.0)  # enforce TE closure
        A_lower = np.random.uniform(-bound, bound, order)
        A_lower = np.append(A_lower, 0.0)

        if not is_valid_airfoil(A_upper, A_lower):
            invalid_samples.append(np.concatenate([A_upper, A_lower]))

    if len(invalid_samples) < n_samples:
        print(f"Warning: Only generated {len(invalid_samples)} invalid samples after {trials} trials.")

    cols = [f"A_u{i}" for i in range(order + 1)] + [f"A_l{i}" for i in range(order + 1)]
    return pd.DataFrame(invalid_samples, columns=cols)

# --- Main function ---
def main():
    csv_path = "cst_fitted_coefficients.csv"  # Your valid CST coefficients file
    df_valid = pd.read_csv(csv_path)
    
    order = 6
    # Enforce TE closure for valid samples
    df_valid[f"A_u{order}"] = 0.0
    df_valid[f"A_l{order}"] = 0.0
    df_valid["valid"] = 1
    
    # Generate invalid samples
    print("Generating invalid CST samples...")
    df_invalid = generate_invalid_samples(n_samples=600, order=order)
    df_invalid["valid"] = 0
    
    # Save invalid samples separately
    df_invalid.to_csv("invalid_cst_samples.csv", index=False)
    print("Invalid samples saved to 'invalid_cst_samples.csv'")
    
    # Combine valid and invalid datasets
    df_training = pd.concat([df_valid, df_invalid], ignore_index=True)
    df_training.to_csv("cst_training_dataset.csv", index=False)
    print("Combined training dataset saved to 'cst_training_dataset.csv'")


    visualize_random_invalid_airfoils("invalid_cst_samples.csv", n_samples=5)
if __name__ == "__main__":
    main()
