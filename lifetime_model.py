import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import lognorm, gaussian_kde
from scipy.signal import find_peaks

# ==========================================
# 1. DATA ANCHORS & CONFIGURATION
# ==========================================
print("Initializing Data Anchors...")

# Anchor 1: Casual Median
# Source: Runner's World 2023 Survey & Shift to Strength study
# "Most runners tend to clock up between 10 and 19 miles every week."
# We target the midpoint: 15 miles/week.
TARGET_MEDIAN_CASUAL = 15 * 52  # 780 miles/year

# Anchor 2: Core Median
# Source: Running USA Global Runner Survey (Skews serious) & Academic Marathon Studies
# "Slower marathon group avg 35km (21mi)/week" vs "Faster group avg 107km (66mi)/week"
# "Sweet spot for serious recreational runners is 50-60 mpw"
# We target a conservative Core median of 35 miles/week (1820 miles/year) 
# to represent the broader "committed" demographic, not just elites.
TARGET_MEDIAN_CORE = 35 * 52    # 1820 miles/year

# Anchor 3: The "Herron Limit"
# Camille Herron: 100k lifetime miles at age 40.
# Implied Max Annual Average over 24 years (16 to 40) = ~4166 miles/yr.
# We treat this as a 4.5-sigma event (1 in ~300,000 for the core population).
HERRON_ANNUAL_MAX = 4166 

# Population Weights (Approximate from Activity Surveys)
# 70% Inactive/Non-Runners
# 24% Casual (Active but low volume)
# 6% Core (High volume/Consistent)
P_NON = 0.70
P_CASUAL = 0.24
P_CORE = 0.06

# Simulation Config
N_SIM = 50000
AGE_RANGE = np.arange(16, 81)

# ==========================================
# 2. HIERARCHICAL OPTIMIZATION (Fitting)
# ==========================================
print(f"Running Optimization (Target Casual={TARGET_MEDIAN_CASUAL}, Core={TARGET_MEDIAN_CORE})...")

def loss_function(params):
    """
    Calculates error between the model's parameters and real-world anchors.
    params: [mu_casual, sigma_casual, mu_core, sigma_core] (Log-Scale)
    """
    mu_c, s_c, mu_s, s_s = params
    
    # 1. Median Constraints
    # LogNormal Median = exp(mu)
    pred_med_c = np.exp(mu_c)
    pred_med_s = np.exp(mu_s)
    
    # Loss is proportional to percentage error
    loss_med_c = ((pred_med_c - TARGET_MEDIAN_CASUAL) / 50)**2
    loss_med_s = ((pred_med_s - TARGET_MEDIAN_CORE) / 50)**2
    
    # 2. The Herron Constraint (Tail Probability)
    # Target Z-score for 4166 miles is 4.5 (Rare event)
    target_z = 4.5
    actual_z = (np.log(HERRON_ANNUAL_MAX) - mu_s) / s_s
    loss_tail = (actual_z - target_z)**2
    
    # Total Loss
    return loss_med_c + loss_med_s + 20 * loss_tail

# Initial Guesses (Log scale)
# ln(780) ~= 6.6, ln(1820) ~= 7.5
init_params = [6.6, 0.8, 7.5, 0.4]

# Run Optimizer
res = minimize(loss_function, init_params, method='Nelder-Mead', tol=1e-4)
best_params = res.x

# Extract Optimized Parameters
OPT_MU_C, OPT_SIG_C, OPT_MU_S, OPT_SIG_S = best_params

print(f"Optimization Complete.")
print(f"  Casual: Median={np.exp(OPT_MU_C):.0f} mi/yr ({np.exp(OPT_MU_C)/52:.1f} mpw), Sigma={OPT_SIG_C:.3f}")
print(f"  Core:   Median={np.exp(OPT_MU_S):.0f} mi/yr ({np.exp(OPT_MU_S)/52:.1f} mpw), Sigma={OPT_SIG_S:.3f}")

# ==========================================
# 3. MONTE CARLO SIMULATION
# ==========================================
print(f"Simulating {N_SIM} Lifetimes...")
np.random.seed(42)

# Assign Groups
groups = np.random.choice([0, 1, 2], size=N_SIM, p=[P_NON, P_CASUAL, P_CORE])
annual_miles = np.zeros((N_SIM, 81)) # Matrix: Person x Age

# --- Process Casual Runners ---
idx_c = np.where(groups == 1)[0]
# Casuals: Start randomly between 18-40. Duration average 10 years.
start_c = np.random.normal(28, 6, len(idx_c))
duration_c = np.random.gamma(2, 5, len(idx_c))
stop_c = start_c + duration_c

# --- Process Core Runners ---
idx_s = np.where(groups == 2)[0]
# Core: Bimodal start (High school starts vs Late onset). Long duration.
is_early = np.random.rand(len(idx_s)) < 0.45
start_s = np.where(is_early, np.random.normal(17, 2, len(idx_s)), np.random.normal(32, 6, len(idx_s)))
duration_s = np.random.gamma(12, 3, len(idx_s)) # Mean ~36 years
stop_s = start_s + duration_s

# --- Fill Annual Miles ---
for age in range(16, 81):
    # Casual Activity (with intermittency noise)
    active_mask_c = (age >= start_c) & (age <= stop_c)
    intermittent_c = np.random.rand(len(idx_c)) < 0.65 # Casuals miss years
    active_idx_c = idx_c[active_mask_c & intermittent_c]
    
    if len(active_idx_c) > 0:
        annual_miles[active_idx_c, age] = np.random.lognormal(OPT_MU_C, OPT_SIG_C, len(active_idx_c))
        
    # Core Activity (High consistency)
    active_mask_s = (age >= start_s) & (age <= stop_s)
    intermittent_s = np.random.rand(len(idx_s)) < 0.90 # Core runners rarely miss years
    active_idx_s = idx_s[active_mask_s & intermittent_s]
    
    if len(active_idx_s) > 0:
        annual_miles[active_idx_s, age] = np.random.lognormal(OPT_MU_S, OPT_SIG_S, len(active_idx_s))

# Calculate Cumulative Lifetime Miles
cumulative = np.cumsum(annual_miles, axis=1)

# ==========================================
# 4. VISUALIZATION & OUTPUT
# ==========================================
print("Generating Visualizations...")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# --- Plot 1: CDF ---
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
ages_to_plot = [30, 50, 70]

for i, age in enumerate(ages_to_plot):
    data = cumulative[:, age]
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, yvals, label=f'Age {age}', color=colors[i+1], linewidth=2.5)

plt.text(100, 0.40, f"Non-Runners (~{int(P_NON*100)}%)\n(Zero Miles)", fontsize=10, 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc'))

plt.xscale('symlog', linthresh=1000)
plt.xlim(-50, 150000)
plt.ylim(0, 1.05)
plt.xlabel("Lifetime Miles (Log Scale)", fontsize=12)
plt.ylabel("Percentile", fontsize=12)
plt.title("CDF of Lifetime Running Miles by Age", fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('output_file_cdf_final.png')
print("Saved CDF to output_file_cdf_final.png")

# --- Plot 2: PDF (Age 60) ---
plt.figure(figsize=(10, 6))
age_60_miles = cumulative[:, 60]
runners_60 = age_60_miles[age_60_miles > 100] # Filter >100 miles

# KDE Plot
kde = gaussian_kde(runners_60)
x_grid = np.linspace(0, 120000, 1000)
density = kde(x_grid)

plt.fill_between(x_grid, density, alpha=0.3, color='#3498db')
plt.plot(x_grid, density, color='#3498db', linewidth=2)

# Annotate Peaks
peaks, _ = find_peaks(density, height=np.max(density)*0.1)
peak_locs = x_grid[peaks]
for peak in peak_locs:
    plt.plot(peak, kde(peak), 'ro')
    plt.text(peak, kde(peak)+0.000002, f"Peak: {int(peak/1000)}k", ha='center', fontweight='bold')

plt.xlabel("Lifetime Miles", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Distribution of Miles for Runners at Age 60", fontsize=14, fontweight='bold')
plt.xlim(0, 120000)
plt.tight_layout()
plt.savefig('output_file_pdf_final.png')
print("Saved PDF to output_file_pdf_final.png")

print("Done.")
