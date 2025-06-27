import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from scipy.special import comb

# ------------------------ CST CORE FUNCTIONS ------------------------
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

# ------------------------ SAMPLE GENERATION ------------------------
def generate_samples_empirical(n_samples, mean_vals, std_vals):
    cols_upper = [c for c in mean_vals.index if c.startswith("A_u")]
    cols_lower = [c for c in mean_vals.index if c.startswith("A_l")]
    samples = []
    for _ in range(n_samples):
        s = pd.Series(np.random.normal(mean_vals, std_vals), index=mean_vals.index)
        s[cols_upper[-1]] = 0.0
        s[cols_lower[-1]] = 0.0
        samples.append(s)
    return pd.DataFrame(samples)

def filter_valid_airfoils(df, model):
    preds = model.predict(df)
    df2 = df.copy()
    df2['valid_pred'] = preds
    return df2[df2['valid_pred']==1].drop(columns=['valid_pred'])

# ------------------------ GEOMETRY CHECKS ------------------------
def reconstruct_geometry(A_u, A_l, n_points=200):
    x = cosine_spacing(n_points)
    y_u = cst_function(x, A_u)
    y_l = cst_function(x, A_l)
    return x[::-1], y_u[::-1], x, y_l

def check_monotonic_x_surface(x, increasing=True, tol=1e-3, tol_edge=1e-3, edge_points=5):
    n=len(x)
    for i in range(n-1):
        diff=x[i+1]-x[i]
        thr = -tol_edge if (i<edge_points or i>n-edge_points-2) else -tol
        if increasing and diff<thr: return False
        if not increasing and diff> -thr: return False
    return True

def check_geometry_crossing(y_u, y_l):
    return np.any((y_u-y_l)[1:-1]<=0)

def check_mid_chord_thickness(x, y_u, y_l, threshold=0.02):
    idx = np.argmin(np.abs(x-0.5))
    return (y_u[idx]-y_l[idx])<threshold

def check_trailing_edge_gap(y_u, y_l, threshold=1e-4):
    return abs(y_u[-1]-y_l[-1])>threshold

def check_smoothness(y, x, window=30, threshold=1.1, ignore_le=20):
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    diffs = np.abs(np.diff(d2y[ignore_le:ignore_le+window]))
    return np.any(diffs>threshold)

def run_geometry_checks_on_sample(A_u, A_l):
    x_u,y_u,x_l,y_l = reconstruct_geometry(A_u,A_l)
    failed=[]
    if not check_monotonic_x_surface(x_u,False): failed.append('Upper surface x not monotonic')
    if not check_monotonic_x_surface(x_l,True): failed.append('Lower surface x not monotonic')
    if check_geometry_crossing(y_u,y_l): failed.append('Geometry crossing detected')
    if check_mid_chord_thickness(x_l,y_u,y_l): failed.append('Thickness below threshold at mid-chord')
    if check_trailing_edge_gap(y_u,y_l): failed.append('Trailing edge gap too large')
    if check_smoothness(y_u,x_u) or check_smoothness(y_l,x_l): failed.append('Surface not smooth')
    return len(failed)==0, failed

def run_geometry_checks_batch(df):
    passed_series = []
    for _,r in df.iterrows():
        A_u = r[[c for c in df.columns if c.startswith('A_u')]].values
        A_l = r[[c for c in df.columns if c.startswith('A_l')]].values
        passed,_ = run_geometry_checks_on_sample(A_u,A_l)
        passed_series.append(passed)
    return pd.Series(passed_series,index=df.index,name='passed')

def plot_failed_airfoils(fail_df, df_all, n=5):
    """
    Plot up to n failed airfoils with reasons in the title.
    """
    print(f'Plotting {n} failed airfoils...')
    for idx in fail_df.index[:n]:
        row = df_all.loc[idx]
        A_u = row[[c for c in df_all.columns if c.startswith('A_u')]].values
        A_l = row[[c for c in df_all.columns if c.startswith('A_l')]].values

        # Reconstruct geometry
        xu, yu_rev, xl, yl = reconstruct_geometry(A_u, A_l)
        y_u = yu_rev[::-1]
        x = xl
        y_l = yl

        # Get failure reasons
        _, reasons = run_geometry_checks_on_sample(A_u, A_l)
        reason_str = '; '.join(reasons) if reasons else 'Unknown reason'

        # Plot airfoil
        plt.figure(figsize=(6, 3))
        plt.plot(xu, yu_rev, 'b-', label='Upper Surface')
        plt.plot(x, y_l, 'r-', label='Lower Surface')
        plt.fill_between(x, y_l, y_u, alpha=0.3)
        plt.title(f'Failed {idx}: {reason_str}')
        plt.legend(loc='upper right')
        plt.axis('equal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def save_airfoil_dat(df, out='generated_airfoils',n=200):
    os.makedirs(out,exist_ok=True)
    for idx,r in df.iterrows():
        A_u = r[[c for c in df.columns if c.startswith('A_u')]].values
        A_l = r[[c for c in df.columns if c.startswith('A_l')]].values
        x=cosine_spacing(n); yu=cst_function(x,A_u); yl=cst_function(x,A_l)
        pts = np.concatenate([np.column_stack((x[::-1],yu[::-1])),np.column_stack((x,yl))])
        name=f"CST_t{int(np.max(yu-yl)*1000)}_mc{int(np.max(np.abs((yu+yl)/2))*1000)}_mlc{int(x[np.argmax(np.abs((yu+yl)/2))]*1000)}_{idx}.dat"
        path=os.path.join(out,name)
        with open(path,'w') as f:
            f.write(name+'\n')
            for xx,yy in pts: f.write(f"{xx:.6f} {yy:.6f}\n")
    print(f"Saved {len(df)} .dat files to '{out}'")

# ------------------------ MAIN ------------------------
if __name__=='__main__':
    base=os.path.dirname(os.path.abspath(__file__))
    df_train=pd.read_csv(os.path.join(base,'cst_training_dataset.csv'))
    df_valid=df_train[df_train['valid']==1]
    cols=[c for c in df_valid.columns if c.startswith('A_u') or c.startswith('A_l')]
    mean_vals=df_valid[cols].mean(); std_vals=df_valid[cols].std()
    print('Computed mean and std dev from valid samples')
    model=joblib.load(os.path.join(base,'gb_cst_classifier_tuned.joblib'))
    print('Generating samples...')
    df_cand=generate_samples_empirical(5000,mean_vals,std_vals)
    print('Filtering valid airfoils...')
    df_valid_ml=filter_valid_airfoils(df_cand,model).reset_index(drop=True)
    print(f"Generated {len(df_valid_ml)} valid samples")
    out_csv=os.path.join(base,'generated_valid_cst_airfoils.csv')
    df_valid_ml.to_csv(out_csv,index=False); print(f"Saved CSV: {out_csv}")
    print('Running checks...')
    results=run_geometry_checks_batch(df_valid_ml)
    failed=results[~results].index; passed=results[results].index
    print(f"Total checked: {len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Passed: {len(passed)}")
    pd.Series(failed).to_csv('failed_airfoils_report.csv',index=False);print('Saved failed report')
    plot_failed_airfoils(results[~results],df_valid_ml,5)
    df_pass=df_valid_ml.loc[passed].reset_index(drop=True)
    save_airfoil_dat(df_pass,'generated_airfoils')
