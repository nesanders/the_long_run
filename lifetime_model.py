import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.ticker as ticker

# ==========================================
# 1. DATA ANCHORS & CONFIGURATION
# ==========================================
print("Initializing Data Anchors...")

# --- DATA SOURCES (DISTRIBUTIONAL SKELETON) ---
# 1. Casual (Strava 12th Annual Year in Sport / Dec 2025)
# Note: Strava highlights "Beginner" (26%) and "Intermediate" (34%) tiers.
# Gen Z median frequency/volume remains low compared to Older cohorts.
DATA_CASUAL_GENZ = 5.0 * 52      # ~260 mi/yr (Reflecting beginner/inconsistent start)
DATA_CASUAL_OLDER = 8.5 * 52     # ~442 mi/yr (Reflecting older casual consistency)

# 2. Recreational (RunRepeat 2025 / Female Anchor)
# RunRepeat 2025 highlights improved finish times (4:51 avg for women).
# We maintain the mpw anchor but align with the 41% female marathoner participation.
DATA_REC_FEMALE = 17.5 * 52      # ~910 mi/yr

# 3. Dedicated (Running USA 2024 / Camille Herron Tail)
# Running USA Global Survey participants are active but not all elite.
DATA_DED_MIXED = 36.5 * 52       # ~1898 mi/yr (Avg for survey participants)
DATA_DED_MALE_MAR = 40.0 * 52    # ~2080 mi/yr (Male Marathoner specific)

# 4. The Herron Anchor (Extreme Tail)
# Camille Herron hit 100,000 miles by Age 40 (April 7, 2022).
# This provides a hard constraint on the possible accumulation for the top 0.001%.
DATA_HERRON_40 = 100000

# 5. Gender Gap (BLS ATUS 2024 results release)
# Men 0.38 hours/day vs Women 0.25 hours/day -> Ratio ~ 1.52
RATIO_GENDER_ATUS = 0.38 / 0.25  
LOG_GENDER_SHIFT_PRIOR = np.log(RATIO_GENDER_ATUS) # ~0.42

# Population Weights (RunRepeat 2025 / Running USA)
# segments remain but individual trajectories are now the focus.
P_NON = 0.60
P_CASUAL = 0.25
P_REC = 0.10
P_DEDICATED = 0.05

# Simulation Config
N_SIM = 50000

# ==========================================
# 2. HIERARCHICAL BAYESIAN MODEL (PYMC)
# ==========================================
print("Building PyMC Model...")

with pm.Model() as lifetime_model:
    # --- 1. Gender Factor ---
    # Log-scale multiplier for male vs female volume
    beta_gender = pm.Normal('beta_gender', mu=LOG_GENDER_SHIFT_PRIOR, sigma=0.1)

    # --- 2. Latent Medians (Female Base - Log Scale) ---
    mu_casual_base = pm.Normal('mu_casual_base', mu=np.log(400), sigma=0.8)
    mu_rec_base = pm.Normal('mu_rec_base', mu=np.log(850), sigma=0.5)
    mu_ded_base = pm.Normal('mu_ded_base', mu=np.log(1800), sigma=0.5)

    # --- 3. Sigmas (Log Scale Widths) ---
    sigma_casual = pm.HalfNormal('sigma_casual', sigma=0.5)
    sigma_rec = pm.HalfNormal('sigma_rec', sigma=0.5)
    sigma_ded = pm.HalfNormal('sigma_ded', sigma=0.4)

    # --- 4. Likelihood / Observations ---
    
    # Casual Observations (Strava Gen Z & Older)
    # Model: mu_mixed = mu_casual_base + 0.5 * beta_gender
    mu_casual_mixed = mu_casual_base + 0.5 * beta_gender
    obs_cas_genz = pm.Normal('obs_cas_genz', mu=mu_casual_mixed, sigma=0.2, observed=np.log(DATA_CASUAL_GENZ))
    obs_cas_older = pm.Normal('obs_cas_older', mu=mu_casual_mixed, sigma=0.2, observed=np.log(DATA_CASUAL_OLDER))
    
    # Recreational Observations
    # 1. Female Anchor (Rec Female / RunRepeat 2025) -> Pure mu_rec_base
    obs_rec_female = pm.Normal('obs_rec_female', mu=mu_rec_base, sigma=0.2, observed=np.log(DATA_REC_FEMALE))
    
    # Dedicated Observations
    # 1. Male Marathoner (RunRepeat 2025) -> mu_ded_base + beta_gender
    obs_ded_male = pm.Normal('obs_ded_male', mu=mu_ded_base + beta_gender, sigma=0.2, observed=np.log(DATA_DED_MALE_MAR))
    
    # 2. Running USA Survey (Mixed) -> mu_ded_base + 0.5 * beta_gender
    obs_ded_mixed = pm.Normal('obs_ded_mixed', mu=mu_ded_base + 0.5 * beta_gender, sigma=0.2, observed=np.log(DATA_DED_MIXED))

    # --- 5. The "Herron Limit" Constraint ---
    # Camille Herron hit 100k by age 40.
    # We model the lifetime total at age 40 for the Dedicated 99.9th percentile.
    # Cumulative = Annual_Base * 24 (assuming start at 16, no decline early)
    # Log(Cumulative) = Log(Annual_Base) + Log(24)
    # The Herron limit is an upper bound on what is humanly possible.
    
    target_z_herron = 3.0 # Top 0.1%
    projected_tail_annual_ded = (mu_ded_base + beta_gender) + target_z_herron * sigma_ded
    lifetime_tail_40 = projected_tail_annual_ded + np.log(24) 
    
    # Constraint: P(Lifetime Total > 100k) should be small for the tail.
    # We set a Potential that penalizes if the tail exceeds 100k.
    pm.Potential('herron_limit_potential', pm.logcdf(pm.Normal.dist(mu=lifetime_tail_40, sigma=0.3), np.log(DATA_HERRON_40)))

# ==========================================
# 3. MCMC SAMPLING
# ==========================================
print("Running MCMC Sampling...")
with lifetime_model:
    idata = pm.sample(draws=1000, tune=500, chains=2, cores=1, return_inferencedata=True, progressbar=False)

summary = az.summary(idata, round_to=3)
print("MCMC Complete. Parameter Summary:")
print(summary)

# Extract Parameters
# Extract Parameters
BETA_GENDER = summary.loc['beta_gender', 'mean']
MU_C_BASE = summary.loc['mu_casual_base', 'mean']
SIG_C = summary.loc['sigma_casual', 'mean']
MU_R_BASE = summary.loc['mu_rec_base', 'mean']
SIG_R = summary.loc['sigma_rec', 'mean']
MU_D_BASE = summary.loc['mu_ded_base', 'mean']
SIG_D = summary.loc['sigma_ded', 'mean']

print(f"\n--- Model Insights ---")
print(f"Gender Multiplier: {np.exp(BETA_GENDER):.2f}x (Males vs Females)")
print(f"Casual Base (F): {np.exp(MU_C_BASE):.0f} mi/yr")
print(f"Rec Base (F):    {np.exp(MU_R_BASE):.0f} mi/yr")
print(f"Ded Base (F):    {np.exp(MU_D_BASE):.0f} mi/yr")

# ==========================================
# 4. SIMULATION (Trajectory-Based)
# ==========================================
print(f"Simulating {N_SIM} Individual Trajectories...")
np.random.seed(42)

def get_decline_factor(age):
    """
    Age-dependent decline curve:
    - Stable until 45
    - ~1.5% annual decline thereafter
    """
    if age <= 45: return 1.0
    return np.exp(-0.015 * (age - 45))

# Groups: 0=Non, 1=Casual, 2=Rec, 3=Ded
groups = np.random.choice([0, 1, 2, 3], size=N_SIM, p=[P_NON, P_CASUAL, P_REC, P_DEDICATED])
annual_miles = np.zeros((N_SIM, 81)) 

# Parameters (Mixed Population for Averages)
MU_C_MIXED = MU_C_BASE + 0.5 * BETA_GENDER
MU_R_MIXED = MU_R_BASE + 0.5 * BETA_GENDER
MU_D_MIXED = MU_D_BASE + 0.5 * BETA_GENDER

# --- Helper to generate lifecycles ---
def generate_lifecycle(n, start_mu, start_std, dur_mu, dur_std):
    starts = np.random.normal(start_mu, start_std, n)
    durs = np.random.gamma(dur_mu, dur_std, n)
    stops = starts + durs
    return starts, stops

# Casual
idx_c = np.where(groups == 1)[0]
starts_c, stops_c = generate_lifecycle(len(idx_c), 28, 6, 2, 5)

# Rec
idx_r = np.where(groups == 2)[0]
starts_r, stops_r = generate_lifecycle(len(idx_r), 24, 5, 8, 4)

# Dedicated
idx_d = np.where(groups == 3)[0]
is_early = np.random.rand(len(idx_d)) < 0.60
starts_d = np.zeros(len(idx_d))
starts_d[is_early] = np.random.normal(16, 2, np.sum(is_early))
starts_d[~is_early] = np.random.normal(30, 5, np.sum(~is_early))
durs_d = np.random.gamma(10, 3.0, len(idx_d)) # ~30 year avg duration
stops_d = starts_d + durs_d

def fill_miles(indices, starts, stops, mu_log, sigma_log, intermittency_prob):
    for age in range(16, 81):
        active_mask = (age >= starts) & (age <= stops)
        is_running_year = np.random.rand(len(indices)) < intermittency_prob
        active_idx = indices[active_mask & is_running_year]
        if len(active_idx) > 0:
            # Apply decline factor
            decline = get_decline_factor(age)
            # Sample log-normal then scale by decline
            annual_miles[active_idx, age] = np.random.lognormal(mu_log, sigma_log, len(active_idx)) * decline

fill_miles(idx_c, starts_c, stops_c, MU_C_MIXED, SIG_C, 0.6)
fill_miles(idx_r, starts_r, stops_r, MU_R_MIXED, SIG_R, 0.8)
fill_miles(idx_d, starts_d, stops_d, MU_D_MIXED, SIG_D, 0.95)

cumulative = np.cumsum(annual_miles, axis=1)

# ==========================================
# 5. EXPORT FOR JS
# ==========================================
print("\n--- JSON FOR INDEX.HTML (Start Copy) ---")
print("const modelData = {")
ages_to_export = [16, 20, 30, 40, 50, 60, 70, 80]
for age in ages_to_export:
    c_active = cumulative[idx_c, age]
    c_active = c_active[c_active > 10]
    r_active = cumulative[idx_r, age]
    r_active = r_active[r_active > 10]
    d_active = cumulative[idx_d, age]
    d_active = d_active[d_active > 10]
    
    def get_stats(arr):
        if len(arr) == 0: return 0, 0
        return np.mean(arr), np.std(arr)

    mc, sc = get_stats(c_active)
    mr, sr = get_stats(r_active)
    md, sd = get_stats(d_active)
    print(f"    {age}: {{ c: [{mc:.0f}, {sc:.0f}], r: [{mr:.0f}, {sr:.0f}], d: [{md:.0f}, {sd:.0f}] }},")

print("};")
print(f"const WEIGHTS = {{ non: {P_NON}, casual: {P_CASUAL}, rec: {P_REC}, ded: {P_DEDICATED} }};")
print(f"const BETA_GENDER = {BETA_GENDER:.2f};")
print("--- JSON FOR INDEX.HTML (End Copy) ---\n")
print("--- JSON FOR INDEX.HTML (End Copy) ---\n")

# ==========================================
# 6. VISUALIZATION (Updated for 3 components)
# ==========================================
print("Generating Visualizations...")
sns.set_style("whitegrid")

# --- PDF Plot (Age 60) ---
plt.figure(figsize=(10, 6))

target_age = 60
# Data for Age 60
d_total = cumulative[:, target_age]
d_c = cumulative[idx_c, target_age]
d_r = cumulative[idx_r, target_age]
d_d = cumulative[idx_d, target_age]

# Filter > 100 for viz
runners_total = d_total[d_total > 100]
runners_c = d_c[d_c > 100]
runners_r = d_r[d_r > 100]
runners_d = d_d[d_d > 100]

x_grid = np.linspace(0, 120000, 1000)

# Weights relative to "All Runners" (cond. on >100)
n_total = len(runners_total)
w_c = len(runners_c) / n_total
w_r = len(runners_r) / n_total
w_d = len(runners_d) / n_total

density_final = np.zeros_like(x_grid)

mode_casual = 0
mode_rec = 0
mode_ded = 0

# Plot components
if len(runners_c) > 50:
    kde = gaussian_kde(runners_c)
    y = kde(x_grid) * w_c
    density_final += y
    plt.fill_between(x_grid, y, alpha=0.2, color='#2ecc71', label='Casual')
    plt.plot(x_grid, y, color='#2ecc71', linestyle='--')
    # Mode label
    im = np.argmax(y)
    mode_casual = x_grid[im]
    plt.text(x_grid[im], y[im], f"Casual\n{int(x_grid[im]/1000)}k", color='#27ae60', ha='center', va='bottom', fontsize=9, fontweight='bold')

if len(runners_r) > 50:
    kde = gaussian_kde(runners_r)
    y = kde(x_grid) * w_r
    density_final += y
    plt.fill_between(x_grid, y, alpha=0.2, color='#f1c40f', label='Recreational')
    plt.plot(x_grid, y, color='#f1c40f', linestyle='--')
    # Mode label
    im = np.argmax(y)
    mode_rec = x_grid[im]
    plt.text(x_grid[im], y[im], f"Rec\n{int(x_grid[im]/1000)}k", color='#f39c12', ha='center', va='bottom', fontsize=9, fontweight='bold')

if len(runners_d) > 50:
    kde = gaussian_kde(runners_d)
    y = kde(x_grid) * w_d
    density_final += y
    plt.fill_between(x_grid, y, alpha=0.2, color='#e74c3c', label='Dedicated')
    plt.plot(x_grid, y, color='#e74c3c', linestyle='--')
    # Mode label
    im = np.argmax(y)
    mode_ded = x_grid[im]
    plt.text(x_grid[im], y[im], f"Ded\n{int(x_grid[im]/1000)}k", color='#c0392b', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Total
plt.plot(x_grid, density_final, color='#34495e', linewidth=2.5, label='Total Distribution')

# Format
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(20000))

plt.xlabel("Lifetime Miles")
plt.ylabel("Density (Cond. on Active)")
plt.title(f"Figure 1: Probability Density of Lifetime Miles (Age {target_age})")
plt.legend()
plt.tight_layout()
plt.savefig('output_file_pdf_final.png')

print(f"\n--- JSON FOR FIGURES (Start Copy) ---")
print("const FIG_STATS = {")
print(f"    age60: {{ casualMode: {int(mode_casual)}, recMode: {int(mode_rec)}, dedMode: {int(mode_ded)} }}")
print("};")
print("--- JSON FOR FIGURES (End Copy) ---\n")

# --- CDF Plot (Age 60) ---
plt.figure(figsize=(10, 6))
# Just plot total CDF for a few ages
# More colorful palette: Green -> Yellow -> Orange -> Red
colors = ['#2ecc71', '#f1c40f', '#3498db', '#e74c3c']
ages_cdf = [30, 50, 70]
for i, age in enumerate(ages_cdf):
    data = cumulative[:, age]
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, yvals, label=f'Age {age}', color=colors[i], linewidth=3)

plt.xscale('symlog', linthresh=1000)
plt.xlim(-50, 150000)
plt.ylim(0, 1.05)
plt.xlabel("Lifetime Miles (Log Scale)")
plt.ylabel("Percentile")
plt.title("Figure 2: Cumulative Distribution Function (CDF)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('output_file_cdf_final.png')


# --- Median Trajectories Plot ---
plt.figure(figsize=(10, 6))

ages = np.arange(16, 81)
medians_c = []
medians_r = []
medians_d = []

# Use vectorized approach from stats is tricky, simpler to use simulation stats
# We can just compute median of active members from simulation at each age
for age in ages:
    # Casual
    c_slice = annual_miles[idx_c, :age].sum(axis=1)
    # Only active? No, median of the GROUP
    medians_c.append(np.median(c_slice))
    
    # Rec
    r_slice = annual_miles[idx_r, :age].sum(axis=1)
    medians_r.append(np.median(r_slice))
    
    # Ded
    d_slice = annual_miles[idx_d, :age].sum(axis=1)
    medians_d.append(np.median(d_slice))

plt.plot(ages, medians_c, label='Casual', color='#27ae60', linewidth=3)
plt.plot(ages, medians_r, label='Recreational', color='#f39c12', linewidth=3)
plt.plot(ages, medians_d, label='Dedicated', color='#c0392b', linewidth=3)

plt.xlabel("Age")
plt.ylabel("Cumulative Miles")
plt.title("Figure 3: Median Accumulated Volume by Group")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.tight_layout()
plt.savefig('output_file_medians_final.png')

print("Done. Saved outputs.")

# --- Consistency / Intermittency Plot ---
print("Generating Consistency Plot...")
plt.figure(figsize=(10, 6))

def get_consistency_ratios(indices, starts, stops):
    ratios = []
    for i, idx in enumerate(indices):
        start_age = int(max(16, starts[i]))
        stop_age = int(min(80, stops[i]))
        duration = stop_age - start_age
        
        if duration < 1:
            continue
            
        # Count active years in this window
        # annual_miles is (N_SIM, 81)
        # We need the specific row for this user: annual_miles[idx]
        user_miles = annual_miles[idx, start_age:stop_age+1]
        active_years = np.sum(user_miles > 0)
        
        ratio = active_years / (duration + 1) # +1 for inclusive
        ratios.append(ratio)
    return np.array(ratios)

ratios_c = get_consistency_ratios(idx_c, starts_c, stops_c)
ratios_r = get_consistency_ratios(idx_r, starts_r, stops_r)
ratios_d = get_consistency_ratios(idx_d, starts_d, stops_d)

sns.kdeplot(ratios_c, color='#2ecc71', fill=True, label='Casual (Simulated)', clip=(0,1))
sns.kdeplot(ratios_r, color='#f1c40f', fill=True, label='Recreational (Simulated)', clip=(0,1))
sns.kdeplot(ratios_d, color='#e74c3c', fill=True, label='Dedicated (Simulated)', clip=(0,1))

plt.xlabel("Consistency Ratio (Active Years / Career Duration)")
plt.ylabel("Density")
plt.title("Figure 4: Consistency of Habit by Group")
plt.xlim(0, 1)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('output_file_consistency.png')
print("Saved output_file_consistency.png")

# --- Plot 5: Annual Volume Distributions (The Inputs) ---
print("Generating Annual Distribution Plot...")
plt.figure(figsize=(10, 6))

x = np.linspace(0, 5000, 1000)

# Casual (Mixed)
dist_c = (np.exp(-(np.log(x) - MU_C_MIXED)**2 / (2 * SIG_C**2)) / (x * SIG_C * np.sqrt(2 * np.pi)))
# Rec (Mixed)
dist_r = (np.exp(-(np.log(x) - MU_R_MIXED)**2 / (2 * SIG_R**2)) / (x * SIG_R * np.sqrt(2 * np.pi)))
# Ded (Mixed)
dist_d = (np.exp(-(np.log(x) - MU_D_MIXED)**2 / (2 * SIG_D**2)) / (x * SIG_D * np.sqrt(2 * np.pi)))

# Normalize peak for visualization comparison (or just show density)
plt.plot(x, dist_c, color='#2ecc71', label='Casual Annual', linewidth=2)
plt.fill_between(x, dist_c, alpha=0.3, color='#2ecc71')

plt.plot(x, dist_r, color='#f1c40f', label='Rec Annual', linewidth=2)
plt.fill_between(x, dist_r, alpha=0.3, color='#f1c40f')

plt.plot(x, dist_d, color='#e74c3c', label='Dedicated Annual', linewidth=2)
plt.fill_between(x, dist_d, alpha=0.3, color='#e74c3c')

plt.xlim(0, 4000)
plt.xlabel("Annual Mileage")
plt.ylabel("Probability Density")
plt.title("Figure 5: The inputs - Annual Volume Distributions")
plt.legend()
plt.tight_layout()
plt.savefig('output_file_annual_dist.png')


# --- Plot 6: Career Gantt Chart ---
print("Generating Gantt Chart...")
plt.figure(figsize=(12, 8))

# Sample 50 ACTIVE runners (ignoring non-runners for the visualization)
n_sample = 50
active_indices = np.where(groups > 0)[0]
if len(active_indices) > n_sample:
    sample_indices = np.random.choice(active_indices, n_sample, replace=False)
else:
    sample_indices = active_indices

# Sort by group then by start age
# We want to group colors together
sorted_sample = []
for g in [1, 2, 3]: # Casual, Rec, Ded
    g_idx = sample_indices[groups[sample_indices] == g]
    # sub sort by duration
    durs = stops_d[np.searchsorted(idx_d, g_idx)] if g == 3 else (stops_r[np.searchsorted(idx_r, g_idx)] if g == 2 else stops_c[np.searchsorted(idx_c, g_idx)])
    # Actually simpler: just get their data from arrays if we mapped them.
    # But simulation arrays are separate.
    # Let's just iterate and build a list
    pass

# Re-build simple list for plotting
plot_data = []
for i in sample_indices:
    g = groups[i]
    if g == 0: continue # Skip non-runners
    
    # Find parameters
    if g == 1:
        color = '#2ecc71'
        label = 'Casual'
        # find in idx_c
        pos = np.where(idx_c == i)[0][0]
        start = starts_c[pos]
        stop = stops_c[pos]
    elif g == 2:
        color = '#f1c40f'
        label = 'Recreational'
        pos = np.where(idx_r == i)[0][0]
        start = starts_r[pos]
        stop = stops_r[pos]
    elif g == 3:
        color = '#e74c3c'
        label = 'Dedicated'
        pos = np.where(idx_d == i)[0][0]
        start = starts_d[pos]
        stop = stops_d[pos]
        
    plot_data.append({
        'id': i, 'group': g, 'color': color, 'start': start, 'stop': stop, 'duration': stop-start
    })

# Sort by group, then duration
plot_data.sort(key=lambda x: (x['group'], x['duration']))

y_pos = 0
for p in plot_data:
    plt.hlines(y=y_pos, xmin=p['start'], xmax=p['stop'], color=p['color'], linewidth=3)
    # Optional: Dots for active years? 
    # Too granular for 50 lines. Just the bars is good.
    y_pos += 1

plt.xlabel("Age")
plt.yticks([])
plt.ylabel("Individual Runners (Sample)")
plt.title("Figure 6: Career Arcs (Sample of 50 Active Runners)")
# Custom Legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='#2ecc71', lw=4),
                Line2D([0], [0], color='#f1c40f', lw=4),
                Line2D([0], [0], color='#e74c3c', lw=4)]
plt.legend(custom_lines, ['Casual', 'Recreational', 'Dedicated'])
plt.xlim(16, 80)
plt.tight_layout()
plt.savefig('output_file_gantt.png')


# --- Plot 7: Persistence Scatter ---
print("Generating Scatter Plot...")
plt.figure(figsize=(10, 6))

n_scatter = 2000
scatter_idx = np.random.choice(N_SIM, n_scatter, replace=False)

x_vals = []
y_vals = []
colors_scatter = []

for i in scatter_idx:
    g = groups[i]
    if g == 0: continue
    
    if g == 1: c = '#2ecc71'
    elif g == 2: c = '#f1c40f'
    elif g == 3: c = '#e74c3c'
    else: c = 'grey'
    
    miles = annual_miles[i]
    active_years = np.sum(miles > 1) 
    total_miles = np.sum(miles)
    
    x_vals.append(active_years)
    y_vals.append(total_miles)
    colors_scatter.append(c)

plt.scatter(x_vals, y_vals, c=colors_scatter, alpha=0.6, s=15, edgecolors='none')
plt.xlabel("Total Active Years")
plt.ylabel("Lifetime Miles")
plt.title("Figure 7: The Persistence Multiplier (Trajectory Clusters)")
plt.legend(custom_lines, ['Casual', 'Recreational', 'Dedicated'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('output_file_scatter.png')

# --- Plot 8: Spaghetti Plot (Individual Trajectories) ---
print("Generating Spaghetti Plot...")
plt.figure(figsize=(10, 6))

n_spaghetti = 100 # Sample paths
spaghetti_idx = np.random.choice(active_indices, min(n_spaghetti, len(active_indices)), replace=False)

for i in spaghetti_idx:
    g = groups[i]
    if g == 1: c = '#2ecc71'
    elif g == 2: c = '#f1c40f'
    elif g == 3: c = '#e74c3c'
    
    # Plot cumulative path
    plt.plot(np.arange(16, 81), cumulative[i, 16:81], color=c, alpha=0.3, linewidth=1)

# Add class medians for reference
plt.plot(np.arange(16, 81), medians_c, color='#1b5e20', linewidth=3, label='Casual Median')
plt.plot(np.arange(16, 81), medians_r, color='#f57f17', linewidth=3, label='Recreational Median')
plt.plot(np.arange(16, 81), medians_d, color='#b71c1c', linewidth=3, label='Dedicated Median')

plt.xlabel("Age")
plt.ylabel("Cumulative Miles")
plt.title("Figure 8: Individual Lifecycles (Spaghetti Plot)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.tight_layout()
plt.savefig('output_file_spaghetti.png')

print("Done. Saved all outputs including output_file_spaghetti.png.")
