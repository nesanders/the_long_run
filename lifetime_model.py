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
# 1. Casual (Strava 2024 mixed)
DATA_CASUAL_GENZ = 6.0 * 52      # ~312 mi/yr
DATA_CASUAL_BOOMER = 9.0 * 52    # ~468 mi/yr

# 2. Recreational (Female Anchor)
# Source: Female <21km racers avg ~16.8 mpw
DATA_REC_FEMALE = 16.8 * 52      # ~874 mi/yr

# 3. Dedicated / Core (Mixed & Male Anchors)
DATA_REC_MAL_MAR = 37.0 * 52     # ~1924 mi/yr (Male Marathoner)
DATA_CORE_USA = 35.0 * 52        # ~1820 mi/yr (Running USA Core - aligned with text)
DATA_COMPETITIVE_TAIL = 50 * 52  # ~2600 mi/yr (Sub-3 Tail)

# 4. Gender Gap (ATUS 2024)
# Men 23 min/day vs Women 15 min/day -> Ratio ~ 1.53
RATIO_GENDER_ATUS = 23.0 / 15.0  
LOG_GENDER_SHIFT_PRIOR = np.log(RATIO_GENDER_ATUS) # ~0.42

# Population Weights (Latent Mixture)
# Expanding to 3 components: Casual, Recreational, Dedicated
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
    # We model a latent "Male Shift" beta_gender. 
    # mu_female = mu_base
    # mu_male = mu_base + beta_gender
    # mu_mixed = mu_base + 0.5 * beta_gender (Assuming rough 50/50 split in mixed surveys)
    
    beta_gender = pm.Normal('beta_gender', mu=LOG_GENDER_SHIFT_PRIOR, sigma=0.1)

    # --- 2. Latent Medians (Female Base - Log Scale) ---
    
    # Casual: Base around 300-500
    mu_casual_base = pm.Normal('mu_casual_base', mu=np.log(300), sigma=0.8)
    
    # Recreational: Base around 800-1000
    mu_rec_base = pm.Normal('mu_rec_base', mu=np.log(800), sigma=0.5)
    
    # Dedicated: Base around 1500+
    mu_ded_base = pm.Normal('mu_ded_base', mu=np.log(1400), sigma=0.5)

    # --- 3. Sigmas (Log Scale Widths) ---
    sigma_casual = pm.HalfNormal('sigma_casual', sigma=0.5)
    sigma_rec = pm.HalfNormal('sigma_rec', sigma=0.5)
    sigma_ded = pm.HalfNormal('sigma_ded', sigma=0.4)

    # --- 4. Likelihood / Observations ---
    
    # Casual Observations (Mixed: Gen Z, Boomer)
    # Model: mu_mixed = mu_casual_base + 0.5 * beta_gender
    mu_casual_mixed = mu_casual_base + 0.5 * beta_gender
    obs_cas_genz = pm.Normal('obs_cas_genz', mu=mu_casual_mixed, sigma=0.2, observed=np.log(DATA_CASUAL_GENZ))
    obs_cas_boom = pm.Normal('obs_cas_boom', mu=mu_casual_mixed, sigma=0.2, observed=np.log(DATA_CASUAL_BOOMER))
    
    # Recreational Observations
    # 1. Female Anchor (Rec Female) -> Pure mu_rec_base
    obs_rec_female = pm.Normal('obs_rec_female', mu=mu_rec_base, sigma=0.2, observed=np.log(DATA_REC_FEMALE))
    
    # Dedicated Observations
    # 1. Male Marathoner -> mu_ded_base + beta_gender
    obs_ded_male = pm.Normal('obs_ded_male', mu=mu_ded_base + beta_gender, sigma=0.2, observed=np.log(DATA_REC_MAL_MAR))
    
    # 2. Running USA Core (Mixed) -> mu_ded_base + 0.5 * beta_gender
    obs_ded_mixed = pm.Normal('obs_ded_mixed', mu=mu_ded_base + 0.5 * beta_gender, sigma=0.2, observed=np.log(DATA_CORE_USA))

    # --- 5. Tail Constraints ---
    # A. Annual Competitiveness
    # 97% of Dedicated Mixed should be around DATA_COMPETITIVE_TAIL (Annual)
    target_z = 2.0 
    projected_tail_annual = (mu_ded_base + 0.5 * beta_gender) + target_z * sigma_ded
    pm.Potential('tail_constraint_annual', pm.logp(pm.Normal.dist(mu=projected_tail_annual, sigma=0.2), np.log(DATA_COMPETITIVE_TAIL)))

    # B. The Herron Limit (Retention / Lifetime)
    # We model the duration (years) for Dedicated runners
    # Prior: Approx 30 years average duration for a "Lifer"
    mu_duration_ded = pm.Normal('mu_duration_ded', mu=30, sigma=5) 
    
    # 40-year-old high-volume runner (Age 16 to 56? No, by Age 40 means 24 years of running)
    # The Herron Constraint usually applies to the EXTREME tail hitting 100k by Age 40.
    # Let's simplify: A "Full Lifetime" (say 50 years) * "Elite Volume" should be bounded.
    # Or, the 99.9th percentile of Lifetime Total shouldn't easily exceed 150k.
    # Let's use the explicit Blog Post claim: "Penalty if tail at Age 40 > 100k"
    # Duration at Age 40 (assuming started at 16) = 24 years.
    # Annual Tail = exp(projected_tail_annual) ~ 2600. 2600 * 24 = 62k. Safe.
    # But if variables drift high, we penalize.
    
    lifetime_tail_40 = projected_tail_annual + np.log(24) # Log Sum = Product
    # We want P(lifetime > 100k) to be small.
    # Soft Soft Constraint: Expect the tail to be BELOW log(100,000)
    pm.Potential('herron_limit', pm.logcdf(pm.Normal.dist(mu=lifetime_tail_40, sigma=0.5), np.log(100000)))

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
BETA_GENDER = summary.loc['beta_gender', 'mean']
MU_C_BASE = summary.loc['mu_casual_base', 'mean']
SIG_C = summary.loc['sigma_casual', 'mean']
MU_R_BASE = summary.loc['mu_rec_base', 'mean']
SIG_R = summary.loc['sigma_rec', 'mean']
MU_D_BASE = summary.loc['mu_ded_base', 'mean']
SIG_D = summary.loc['sigma_ded', 'mean']
DUR_D_MEAN = summary.loc['mu_duration_ded', 'mean']

print(f"\n--- Model Insights ---")
print(f"Gender Multiplier: {np.exp(BETA_GENDER):.2f}x (Males vs Females)")
print(f"Casual Base (F): {np.exp(MU_C_BASE):.0f} mi/yr")
print(f"Rec Base (F):    {np.exp(MU_R_BASE):.0f} mi/yr")
print(f"Ded Base (F):    {np.exp(MU_D_BASE):.0f} mi/yr")
print(f"Ded Duration:    {DUR_D_MEAN:.1f} yrs (avg)")

# ==========================================
# 4. SIMULATION (For Visuals)
# ==========================================
# We simulate a "Mixed" population for the static charts
# Mixed Mu = Base + 0.5 * Beta

print(f"Simulating {N_SIM} Mixed Lifetimes...")
np.random.seed(42)

# Groups: 0=Non, 1=Casual, 2=Rec, 3=Ded
groups = np.random.choice([0, 1, 2, 3], size=N_SIM, p=[P_NON, P_CASUAL, P_REC, P_DEDICATED])
annual_miles = np.zeros((N_SIM, 81)) 

# Parameters (Mixed)
MU_C_MIXED = MU_C_BASE + 0.5 * BETA_GENDER
MU_R_MIXED = MU_R_BASE + 0.5 * BETA_GENDER
MU_D_MIXED = MU_D_BASE + 0.5 * BETA_GENDER

# --- Helper to generate lifecycles ---
def generate_lifecycle(n, start_mu, start_std, dur_mu, dur_std, miles_mu, miles_sig, intermittency):
    starts = np.random.normal(start_mu, start_std, n)
    durs = np.random.gamma(dur_mu, dur_std, n)
    stops = starts + durs
    return starts, stops

# Casual
idx_c = np.where(groups == 1)[0]
starts_c, stops_c = generate_lifecycle(len(idx_c), 28, 6, 2, 5, MU_C_MIXED, SIG_C, 0.65)

# Rec
idx_r = np.where(groups == 2)[0]
starts_r, stops_r = generate_lifecycle(len(idx_r), 24, 5, 8, 4, MU_R_MIXED, SIG_R, 0.80)

# Dedicated
idx_d = np.where(groups == 3)[0]
is_early = np.random.rand(len(idx_d)) < 0.60
starts_d_early = np.random.normal(16, 2, np.sum(is_early))
starts_d_late = np.random.normal(30, 5, np.sum(~is_early))
starts_d = np.concatenate([starts_d_early, starts_d_late]) # Naive concat, random order doesn't matter for idx
# fix order
starts_d = np.zeros(len(idx_d))
starts_d[is_early] = starts_d_early
starts_d[~is_early] = starts_d_late

# Use inferred duration from PyMC
# DUR_D_MEAN ~ 30. Let's use Gamma(k, theta) where k*theta = mean.
# shape=10 gives reasonable variance. theta = mean/10.
durs_d = np.random.gamma(10, DUR_D_MEAN/10.0, len(idx_d))
stops_d = starts_d + durs_d

def fill_miles(indices, starts, stops, mu_log, sigma_log, intermittency_prob):
    for age in range(16, 81):
        active_mask = (age >= starts) & (age <= stops)
        # Intermittency check (annual probability of running given active phase)
        is_running_year = np.random.rand(len(indices)) < intermittency_prob
        active_idx = indices[active_mask & is_running_year]
        if len(active_idx) > 0:
            annual_miles[active_idx, age] = np.random.lognormal(mu_log, sigma_log, len(active_idx))

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
    # Get cumulative miles for each group at this age
    c_vals = cumulative[idx_c, age]
    r_vals = cumulative[idx_r, age]
    d_vals = cumulative[idx_d, age]
    
    # Filter for "Active" (having some minimal mileage to fit the log-normal)
    # Since we use weights in the JS, standard practice is to model the distribution of the *group* 
    # but the log-normal fit really only works on the positive values.
    # The JS `logNormCDF` assumes the input (Mean/Std) matches the data.
    
    # We'll compute the mean/std of the POSITIVE (>10 miles) members of each group.
    # The "Intermittency" or "Non-Start" is handled by the group weights (Pi) 
    # OR we can assume the group is "conditional on being a runner of that type".
    # In the JS, we sum: W_C * CDF_C + W_R * CDF_R + W_D * CDF_D
    # So CDF_C should represent the distribution of a random Casual runner.
    # If a Casual runner has 0 miles at age 30 (not started), they are at 0.
    # SimpleMean/Std works best if we treat the "Active" distribution.
    
    c_active = c_vals[c_vals > 10]
    r_active = r_vals[r_vals > 10]
    d_active = d_vals[d_vals > 10] # Dedicated rarely 0 if started, but safety first
    
    # helper
    def get_stats(arr):
        if len(arr) == 0: return 0, 0
        return np.mean(arr), np.std(arr)

    mc, sc = get_stats(c_active)
    mr, sr = get_stats(r_active)
    md, sd = get_stats(d_active)

    print(f"    {age}: {{ c: [{mc:.0f}, {sc:.0f}], r: [{mr:.0f}, {sr:.0f}], d: [{md:.0f}, {sd:.0f}] }},")

print("};")
print(f"const WEIGHTS = {{ non: {P_NON}, casual: {P_CASUAL}, rec: {P_REC}, ded: {P_DEDICATED} }};")
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
plt.title(f"Mixture Distribution of Lifetime Miles (Age {target_age})")
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
ages_cdf = [30, 50, 60, 70] # added 60 to have 4 lines if we want, or keep 3. Let's do 30, 50, 70.
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
plt.title("CDF of Lifetime Running Miles by Age")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('output_file_cdf_final.png')


# --- Median Trajectories Plot ---
plt.figure(figsize=(10, 6))

ages = np.arange(16, 81)
med_c = []
med_r = []
med_d = []

for age in ages:
    med_c.append(np.median(cumulative[idx_c, age]))
    med_r.append(np.median(cumulative[idx_r, age]))
    med_d.append(np.median(cumulative[idx_d, age]))

plt.plot(ages, med_c, label='Casual', color='#2ecc71', linewidth=2.5)
plt.plot(ages, med_r, label='Recreational', color='#f1c40f', linewidth=2.5)
plt.plot(ages, med_d, label='Dedicated', color='#e74c3c', linewidth=2.5)

ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.xlabel("Age")
plt.ylabel("Cumulative Miles")
plt.title("Median Lifetime Miles by Age (Three Tribes)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('output_file_medians_final.png')

print("Done. Saved outputs.")
