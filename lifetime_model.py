import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ==========================================
# 1. DATA ANCHORS & CONFIGURATION
# ==========================================
print("Initializing Data Anchors...")

# --- DATA SOURCES ---
# 1. Runner's World / Historical Anchor: 15 miles/week
#    Source: RW, general surveys.
DATA_CASUAL_RW = 15 * 52  # 780 miles/year

# 2. Strava Year in Sport 2024: 8.1 miles/week (Median)
#    Source: Strava 2024 Report.
DATA_CASUAL_STRAVA = 8.1 * 52  # ~421 miles/year

# 3. Core Median Anchor: 35 miles/week
#    Source: Running USA, committed runner studies.
DATA_CORE_RUNNING_USA = 35 * 52  # 1820 miles/year

# 4. The "Herron Limit" (Max possible)
#    Source: Camille Herron.
HERRON_ANNUAL_MAX = 4166

# Population Weights
P_NON = 0.70
P_CASUAL = 0.24
P_CORE = 0.06

# Simulation Config
N_SIM = 50000

# ==========================================
# 2. HIERARCHICAL BAYESIAN MODEL (PYMC)
# ==========================================
# We treat the "Casual" and "Core" medians as latent variables.
# We have "observations" for these medians from our data sources.
print("Building PyMC Model...")

with pm.Model() as lifetime_model:
    # --- Priors for Latent Medians (Log Scale) ---
    # mu_casual: The "true" median of the casual population.
    # We set a broad prior roughly between 300 and 1000 miles/year.
    mu_casual_log = pm.Normal('mu_casual_log', mu=np.log(600), sigma=1.0)
    
    # mu_core: The "true" median of the core population.
    # We set a broad prior roughly around 1500-2500 miles/year.
    mu_core_log = pm.Normal('mu_core_log', mu=np.log(1800), sigma=1.0)

    # --- Priors for Population Standard Deviations (Log Scale) ---
    # Core runners are likely more consistent (lower sigma) than casuals.
    sigma_casual = pm.HalfNormal('sigma_casual', sigma=0.5)
    sigma_core = pm.HalfNormal('sigma_core', sigma=0.5)

    # --- Likelihood (Data Observations) ---
    # We observe estimated medians from different sources. 
    # We model these observations as coming from the true latent median + some observation noise.
    
    # Casual Observations
    obs_casual_rw = pm.Normal('obs_casual_rw', mu=mu_casual_log, sigma=0.2, observed=np.log(DATA_CASUAL_RW))
    obs_casual_strava = pm.Normal('obs_casual_strava', mu=mu_casual_log, sigma=0.2, observed=np.log(DATA_CASUAL_STRAVA))
    
    # Core Observation
    obs_core_rusa = pm.Normal('obs_core_rusa', mu=mu_core_log, sigma=0.2, observed=np.log(DATA_CORE_RUNNING_USA))

    # --- Constraints (Herron Limit) ---
    # Instead of a hard potential, we can treat this as a soft constraint or just a check.
    # Here, we keep it as a potential to ensure the tail doesn't go too wild, 
    # effectively saying "The 4.5 sigma event should be around HERRON_ANNUAL_MAX".
    target_z = 4.5
    projected_max = mu_core_log + target_z * sigma_core
    # We want projected_max to be close to log(HERRON_ANNUAL_MAX)
    pm.Potential('herron_constraint', pm.logp(pm.Normal.dist(mu=projected_max, sigma=0.2), np.log(HERRON_ANNUAL_MAX)))

# ==========================================
# 3. MCMC SAMPLING
# ==========================================
print("Running MCMC Sampling...")
with lifetime_model:
    # Reduced tuning/draws for speed as requested ("within a few seconds")
    # NUTS is efficient enough that 1000/500 is often plenty for this simple geometry.
    idata = pm.sample(draws=1000, tune=500, chains=2, cores=1, return_inferencedata=True, progressbar=False)

summary = az.summary(idata, round_to=3)
print("MCMC Complete. Parameter Summary:")
print(summary)

# Extract Means
OPT_MU_C = summary.loc['mu_casual_log', 'mean']
OPT_SIG_C = summary.loc['sigma_casual', 'mean']
OPT_MU_S = summary.loc['mu_core_log', 'mean']
OPT_SIG_S = summary.loc['sigma_core', 'mean']

print(f"Posterior Means:")
print(f"  Casual Median: {np.exp(OPT_MU_C):.0f} mi/yr (approx {np.exp(OPT_MU_C)/52:.1f} mpw)")
print(f"  Core Median:   {np.exp(OPT_MU_S):.0f} mi/yr (approx {np.exp(OPT_MU_S)/52:.1f} mpw)")

# ==========================================
# 4. MONTE CARLO SIMULATION
# ==========================================
print(f"Simulating {N_SIM} Lifetimes...")
np.random.seed(42)

groups = np.random.choice([0, 1, 2], size=N_SIM, p=[P_NON, P_CASUAL, P_CORE])
annual_miles = np.zeros((N_SIM, 81)) 

# Casuals
idx_c = np.where(groups == 1)[0]
start_c = np.random.normal(28, 6, len(idx_c))
duration_c = np.random.gamma(2, 5, len(idx_c))
stop_c = start_c + duration_c

# Core
idx_s = np.where(groups == 2)[0]
is_early = np.random.rand(len(idx_s)) < 0.45
start_s = np.where(is_early, np.random.normal(17, 2, len(idx_s)), np.random.normal(32, 6, len(idx_s)))
duration_s = np.random.gamma(12, 3, len(idx_s))
stop_s = start_s + duration_s

for age in range(16, 81):
    # Casual
    active_mask_c = (age >= start_c) & (age <= stop_c)
    intermittent_c = np.random.rand(len(idx_c)) < 0.65 
    active_idx_c = idx_c[active_mask_c & intermittent_c]
    if len(active_idx_c) > 0:
        annual_miles[active_idx_c, age] = np.random.lognormal(OPT_MU_C, OPT_SIG_C, len(active_idx_c))
        
    # Core
    active_mask_s = (age >= start_s) & (age <= stop_s)
    intermittent_s = np.random.rand(len(idx_s)) < 0.90 
    active_idx_s = idx_s[active_mask_s & intermittent_s]
    if len(active_idx_s) > 0:
        annual_miles[active_idx_s, age] = np.random.lognormal(OPT_MU_S, OPT_SIG_S, len(active_idx_s))

cumulative = np.cumsum(annual_miles, axis=1)

# ==========================================
# 5. EXPORT FOR JS
# ==========================================
print("\n--- JSON FOR INDEX.HTML ---")
ages_to_export = [16, 20, 30, 40, 50, 60, 70, 80]
print("const modelData = {")
for age in ages_to_export:
    # Casual Stats at this age
    # Filter for those who have started by this age to avoid zero-bias if we want 'active' dist,
    # but the JS model likely expects the distribution of the *group* (Casuals) including those who haven't started or have stopped?
    # The JS uses logNormCDF, which implies > 0.
    # We should look at the non-zero values for the group at that age?
    # Or just the stats of the cumulative total for specific groups?
    # Index.html uses: cdf_c = logNormCDF(miles, p.c[0], p.c[1])
    # This implies p.c are parameters of the LogNormal distribution of cumulative miles.
    # LogNormal params are mu and sigma (log-scale).
    # But the JS `logNormCDF` function takes `mean_linear` and `std_linear` and converts them!
    # "function logNormCDF(x, mean_linear, std_linear)... const mu_log = ..."
    # So we need the Mean and Std of the linear data (cumulative miles).
    
    c_vals = cumulative[idx_c, age]
    s_vals = cumulative[idx_s, age]
    
    # Filter > 0 to fit the LogNormal assumption of the JS widget
    c_vals = c_vals[c_vals > 10]
    s_vals = s_vals[s_vals > 10]
    
    if len(c_vals) == 0:
        c_mean, c_std = 0, 0
    else:
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        
    if len(s_vals) == 0:
        s_mean, s_std = 0, 0
    else:
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)

    print(f"    {age}: {{ c: [{c_mean:.0f}, {c_std:.0f}], s: [{s_mean:.0f}, {s_std:.0f}] }},")
print("};")
print("---------------------------\n")

# ==========================================
# 6. VISUALIZATION
# ==========================================
print("Generating Visualizations...")
sns.set_style("whitegrid")

# CDF
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
ages_to_plot = [30, 50, 70]
for i, age in enumerate(ages_to_plot):
    data = cumulative[:, age]
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    plt.plot(sorted_data, yvals, label=f'Age {age}', color=colors[i+1], linewidth=2.5)

plt.xscale('symlog', linthresh=1000)
plt.xlim(-50, 150000)
plt.ylim(0, 1.05)
plt.xlabel("Lifetime Miles (Log Scale)")
plt.ylabel("Percentile")
plt.title("CDF of Lifetime Running Miles by Age")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('output_file_cdf_final.png')

# PDF (Age 60)
plt.figure(figsize=(10, 6))

# Extract data for Age 60
all_miles_60 = cumulative[:, 60]
casual_miles_60 = cumulative[idx_c, 60]
core_miles_60 = cumulative[idx_s, 60]

# Filter to active runners (> 100 miles) for visualization
runners_60 = all_miles_60[all_miles_60 > 100]
casual_runners_60 = casual_miles_60[casual_miles_60 > 100]
core_runners_60 = core_miles_60[core_miles_60 > 100]

# Define grid
x_grid = np.linspace(0, 120000, 1000)

# 1. Statistics for weighting
prop_casual_among_runners = len(casual_runners_60) / len(runners_60)
prop_core_among_runners = len(core_runners_60) / len(runners_60)

# Initialize densities
density_casual_scaled = np.zeros_like(x_grid)
density_core_scaled = np.zeros_like(x_grid)

# Casual Component
if len(casual_runners_60) > 10:
    # Use a slightly tighter bandwidth for visualization if needed, but default is usually fine
    kde_casual = gaussian_kde(casual_runners_60)
    density_casual_scaled = kde_casual(x_grid) * prop_casual_among_runners
    plt.fill_between(x_grid, density_casual_scaled, alpha=0.2, color='#2ecc71', label='Casual Component')
    plt.plot(x_grid, density_casual_scaled, color='#2ecc71', linestyle='--')

# Core Component
if len(core_runners_60) > 10:
    kde_core = gaussian_kde(core_runners_60)
    density_core_scaled = kde_core(x_grid) * prop_core_among_runners
    plt.fill_between(x_grid, density_core_scaled, alpha=0.2, color='#e74c3c', label='Core Component')
    plt.plot(x_grid, density_core_scaled, color='#e74c3c', linestyle='--')

# Total Density (Sum of components)
density_total_sum = density_casual_scaled + density_core_scaled
plt.plot(x_grid, density_total_sum, color='#34495e', linewidth=2.5, label='Total Density')

# Annotate Mode of Casual Component
if len(casual_runners_60) > 10:
    idx_max_casual = np.argmax(density_casual_scaled)
    peak_casual = x_grid[idx_max_casual]
    val_casual = density_casual_scaled[idx_max_casual]
    # Mark the peak
    plt.plot(peak_casual, val_casual, 'o', color='#2ecc71')
    plt.text(peak_casual, val_casual + 0.000002, f"Casual Mode\n{int(peak_casual):,} mi", 
             ha='right', va='bottom', fontweight='bold', color='#27ae60', fontsize=10)

# Annotate Mode of Core Component
if len(core_runners_60) > 10:
    idx_max_core = np.argmax(density_core_scaled)
    peak_core = x_grid[idx_max_core]
    val_core = density_core_scaled[idx_max_core]
    # Mark the peak
    plt.plot(peak_core, val_core, 'o', color='#c0392b')
    plt.text(peak_core, val_core + 0.000002, f"Core Mode\n{int(peak_core):,} mi", 
             ha='center', va='bottom', fontweight='bold', color='#c0392b', fontsize=10)

# X-Axis Formatting
import matplotlib.ticker as ticker
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(20000))

plt.xlabel("Lifetime Miles")
plt.ylabel("Density (Cond. on Active)")
plt.title("Mixture Distribution of Lifetime Miles (Age 60)")
plt.legend()
plt.tight_layout()
plt.savefig('output_file_pdf_final.png')

print("Done. Saved outputs.")
