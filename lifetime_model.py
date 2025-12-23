import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy.stats import gaussian_kde
import matplotlib.ticker as ticker
import json
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================

class Config:
    """Central configuration for data anchors and simulation parameters."""
    # --- DATA SOURCES (DISTRIBUTIONAL SKELETON) ---
    DATA_CASUAL_GENZ = 5.0 * 52      # ~260 mi/yr
    DATA_CASUAL_OLDER = 8.5 * 52     # ~442 mi/yr
    DATA_REC_FEMALE = 17.5 * 52      # ~910 mi/yr
    DATA_DED_MIXED = 36.5 * 52       # ~1898 mi/yr
    DATA_DED_MALE_MAR = 40.0 * 52    # ~2080 mi/yr
    DATA_HERRON_40 = 100000

    # Gender Gap Ratio
    RATIO_GENDER_ATUS = 0.38 / 0.25  
    LOG_GENDER_SHIFT_PRIOR = np.log(RATIO_GENDER_ATUS) # ~0.42

    # Population Weights
    P_NON = 0.60
    P_CASUAL = 0.25
    P_REC = 0.10
    P_DEDICATED = 0.05

    # Simulation Config
    N_SIM = 50000
    SEED = 42

def build_lifetime_model(config):
    """Builds and returns the PyMC hierarchical model."""
    with pm.Model() as model:
        # --- 1. Gender Factor ---
        beta_gender = pm.Normal('beta_gender', mu=config.LOG_GENDER_SHIFT_PRIOR, sigma=0.1)

        # --- 2. Latent Medians (Female Base - Log Scale) ---
        mu_casual_base = pm.Normal('mu_casual_base', mu=np.log(400), sigma=0.8)
        mu_rec_base = pm.Normal('mu_rec_base', mu=np.log(850), sigma=0.5)
        mu_ded_base = pm.Normal('mu_ded_base', mu=np.log(1800), sigma=0.5)

        # --- 3. Sigmas (Log Scale Widths) ---
        sigma_casual = pm.HalfNormal('sigma_casual', sigma=0.5)
        sigma_rec = pm.HalfNormal('sigma_rec', sigma=0.5)
        sigma_ded = pm.HalfNormal('sigma_ded', sigma=0.5)

        # --- 4. Likelihood / Observations ---
        mu_casual_mixed = mu_casual_base + 0.5 * beta_gender
        pm.Normal('obs_cas_genz', mu=mu_casual_mixed, sigma=0.2, observed=np.array([np.log(config.DATA_CASUAL_GENZ)]))
        pm.Normal('obs_cas_older', mu=mu_casual_mixed, sigma=0.2, observed=np.array([np.log(config.DATA_CASUAL_OLDER)]))
        
        pm.Normal('obs_rec_female', mu=mu_rec_base, sigma=0.2, observed=np.array([np.log(config.DATA_REC_FEMALE)]))
        
        pm.Normal('obs_ded_male', mu=mu_ded_base + beta_gender, sigma=0.2, observed=np.array([np.log(config.DATA_DED_MALE_MAR)]))
        pm.Normal('obs_ded_mixed', mu=mu_ded_base + 0.5 * beta_gender, sigma=0.2, observed=np.array([np.log(config.DATA_DED_MIXED)]))

        # --- 5. The "Herron Limit" Constraint ---
        # NOTE: Camille Herron reached 100k by age 40. She is female, so the 
        # constraint is applied to the female base (mu_ded_base) on the extreme tail.
        target_z_herron = 3.0 # Top 0.1%
        projected_tail_annual_ded = mu_ded_base + target_z_herron * sigma_ded
        lifetime_tail_40 = projected_tail_annual_ded + np.log(24) 
        
        pm.Potential('herron_limit_potential', pm.logcdf(pm.Normal.dist(mu=lifetime_tail_40, sigma=0.3), np.log(config.DATA_HERRON_40)))
    
    return model

def run_mcmc(config, draws=5000, tune=1000):
    """Runs MCMC sampling and returns inference data."""
    print("Building and sampling PyMC Model...")
    model = build_lifetime_model(config)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1, return_inferencedata=True, progressbar=False)
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=False)
    
    summary = az.summary(idata, round_to=3)
    print("\nMCMC Complete. Parameter Summary:")
    print(summary)
    
    params = {
        'beta_gender': summary.loc['beta_gender', 'mean'],
        'mu_c_base': summary.loc['mu_casual_base', 'mean'],
        'sig_c': summary.loc['sigma_casual', 'mean'],
        'mu_r_base': summary.loc['mu_rec_base', 'mean'],
        'sig_r': summary.loc['sigma_rec', 'mean'],
        'mu_d_base': summary.loc['mu_ded_base', 'mean'],
        'sig_d': summary.loc['sigma_ded', 'mean'],
    }
    
    print(f"\n--- Model Insights ---")
    print(f"Gender Multiplier: {np.exp(params['beta_gender']):.2f}x (Males vs Females)")
    print(f"Casual Base (F): {np.exp(params['mu_c_base']):.0f} mi/yr")
    print(f"Rec Base (F):    {np.exp(params['mu_r_base']):.0f} mi/yr")
    print(f"Ded Base (F):    {np.exp(params['mu_d_base']):.0f} mi/yr")
    
    return idata, params

# ==========================================
# 2. SIMULATION (Trajectory-Based)
# ==========================================

def get_decline_factor(age):
    """Age-dependent decline curve: Stable until 45, ~1.5% annual decline thereafter."""
    if age <= 45: return 1.0
    return np.exp(-0.015 * (age - 45))

def generate_lifecycle(n, start_mu, start_std, dur_mu, dur_std):
    """Generates career start ages and durations."""
    starts = np.random.normal(start_mu, start_std, n)
    durs = np.random.gamma(dur_mu, dur_std, n)
    stops = starts + durs
    return starts, stops

def fill_miles(annual_miles, indices, starts, stops, mu_log, sigma_log, intermittency_prob):
    """Fills the annual_miles matrix for a given cohort."""
    for age in range(16, 81):
        active_mask = (age >= starts) & (age <= stops)
        is_running_year = np.random.rand(len(indices)) < intermittency_prob
        active_idx = indices[active_mask & is_running_year]
        if len(active_idx) > 0:
            decline = get_decline_factor(age)
            annual_miles[active_idx, age] = np.random.lognormal(mu_log, sigma_log, len(active_idx)) * decline

def run_simulation(config, params):
    """Simulates individual trajectories based on model parameters."""
    print(f"Simulating {config.N_SIM} Individual Trajectories...")
    np.random.seed(config.SEED)

    # Groups: 0=Non, 1=Casual, 2=Rec, 3=Ded
    groups = np.random.choice([0, 1, 2, 3], size=config.N_SIM, p=[config.P_NON, config.P_CASUAL, config.P_REC, config.P_DEDICATED])
    annual_miles = np.zeros((config.N_SIM, 81)) 

    # Parameters (Mixed Population for Averages)
    mu_c_mixed = params['mu_c_base'] + 0.5 * params['beta_gender']
    mu_r_mixed = params['mu_r_base'] + 0.5 * params['beta_gender']
    mu_d_mixed = params['mu_d_base'] + 0.5 * params['beta_gender']

    # Casual
    idx_c = np.where(groups == 1)[0]
    starts_c, stops_c = generate_lifecycle(len(idx_c), 28, 6, 2, 5)
    fill_miles(annual_miles, idx_c, starts_c, stops_c, mu_c_mixed, params['sig_c'], 0.6)

    # Rec
    idx_r = np.where(groups == 2)[0]
    starts_r, stops_r = generate_lifecycle(len(idx_r), 24, 5, 8, 4)
    fill_miles(annual_miles, idx_r, starts_r, stops_r, mu_r_mixed, params['sig_r'], 0.8)

    # Dedicated
    idx_d = np.where(groups == 3)[0]
    is_early = np.random.rand(len(idx_d)) < 0.60
    starts_d = np.zeros(len(idx_d))
    starts_d[is_early] = np.random.normal(16, 2, np.sum(is_early))
    starts_d[~is_early] = np.random.normal(30, 5, np.sum(~is_early))
    durs_d = np.random.gamma(10, 3.0, len(idx_d)) 
    stops_d = starts_d + durs_d
    fill_miles(annual_miles, idx_d, starts_d, stops_d, mu_d_mixed, params['sig_d'], 0.95)

    cumulative = np.cumsum(annual_miles, axis=1)
    
    simulation_results = {
        'groups': groups,
        'annual_miles': annual_miles,
        'cumulative': cumulative,
        'cohort_indices': {
            'c': idx_c, 'r': idx_r, 'd': idx_d
        },
        'lifecycle_data': {
            'c': (starts_c, stops_c),
            'r': (starts_r, stops_r),
            'd': (starts_d, stops_d)
        }
    }
    
    return simulation_results, mu_c_mixed, mu_r_mixed, mu_d_mixed

# ==========================================
# 3. VISUALIZATION
# ==========================================

def plot_pdf(config, sim_results, target_age=60):
    """Generates Figure 1: Probability Density of Lifetime Miles."""
    plt.figure(figsize=(10, 6))
    
    cumulative = sim_results['cumulative']
    idx_c = sim_results['cohort_indices']['c']
    idx_r = sim_results['cohort_indices']['r']
    idx_d = sim_results['cohort_indices']['d']
    
    d_total = cumulative[:, target_age]
    d_c = cumulative[idx_c, target_age]
    d_r = cumulative[idx_r, target_age]
    d_d = cumulative[idx_d, target_age]

    runners_total = d_total[d_total > 100]
    runners_c = d_c[d_c > 100]
    runners_r = d_r[d_r > 100]
    runners_d = d_d[d_d > 100]

    x_grid = np.linspace(0, 120000, 1000)
    n_total = len(runners_total)
    
    density_final = np.zeros_like(x_grid)
    modes = {}

    components = [
        (runners_c, '#2ecc71', '#27ae60', 'Casual'),
        (runners_r, '#f1c40f', '#f39c12', 'Recreational'),
        (runners_d, '#e74c3c', '#c0392b', 'Dedicated')
    ]

    for data, color, text_color, label in components:
        if len(data) > 50:
            weight = len(data) / n_total
            kde = gaussian_kde(data)
            y = kde(x_grid) * weight
            density_final += y
            plt.fill_between(x_grid, y, alpha=0.2, color=color, label=label)
            plt.plot(x_grid, y, color=color, linestyle='--')
            
            im = np.argmax(y)
            modes[label.lower()] = int(x_grid[im])
            plt.text(x_grid[im], y[im], f"{label[:3]}\n{int(x_grid[im]/1000)}k", 
                     color=text_color, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.plot(x_grid, density_final, color='#34495e', linewidth=2.5, label='Total Distribution')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20000))

    plt.xlabel("Lifetime Miles")
    plt.ylabel("Density (Cond. on Active)")
    plt.title(f"Figure 1: Probability Density of Lifetime Miles (Age {target_age})")
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_file_pdf_final.png')
    
    return modes

def plot_cdf(sim_results):
    """Generates Figure 2: Cumulative Distribution Function (CDF)."""
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#f1c40f', '#3498db', '#e74c3c']
    ages_cdf = [30, 50, 70]
    cumulative = sim_results['cumulative']
    
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

def plot_medians(sim_results):
    """Generates Figure 3: Median Accumulated Volume by Group."""
    plt.figure(figsize=(10, 6))
    ages = np.arange(16, 81)
    
    annual_miles = sim_results['annual_miles']
    idx_c = sim_results['cohort_indices']['c']
    idx_r = sim_results['cohort_indices']['r']
    idx_d = sim_results['cohort_indices']['d']
    
    medians = {'c': [], 'r': [], 'd': []}
    for age in ages:
        medians['c'].append(np.median(annual_miles[idx_c, :age].sum(axis=1)))
        medians['r'].append(np.median(annual_miles[idx_r, :age].sum(axis=1)))
        medians['d'].append(np.median(annual_miles[idx_d, :age].sum(axis=1)))

    plt.plot(ages, medians['c'], label='Casual', color='#27ae60', linewidth=3)
    plt.plot(ages, medians['r'], label='Recreational', color='#f39c12', linewidth=3)
    plt.plot(ages, medians['d'], label='Dedicated', color='#c0392b', linewidth=3)

    plt.xlabel("Age")
    plt.ylabel("Cumulative Miles")
    plt.title("Figure 3: Median Accumulated Volume by Group")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
    plt.tight_layout()
    plt.savefig('output_file_medians_final.png')
    
    return medians

def plot_consistency(sim_results):
    """Generates Figure 4: Consistency of Habit by Group."""
    plt.figure(figsize=(10, 6))
    
    def get_consistency_ratios(indices, starts, stops, annual_miles):
        ratios = []
        for i, idx in enumerate(indices):
            start_age = int(max(16, starts[i]))
            stop_age = int(min(80, stops[i]))
            duration = stop_age - start_age
            if duration < 1: continue
            user_miles = annual_miles[idx, start_age:stop_age+1]
            ratio = np.sum(user_miles > 0) / (duration + 1)
            ratios.append(ratio)
        return np.array(ratios)

    annual_miles = sim_results['annual_miles']
    cohort_idx = sim_results['cohort_indices']
    lifecycle = sim_results['lifecycle_data']
    
    ratios_c = get_consistency_ratios(cohort_idx['c'], lifecycle['c'][0], lifecycle['c'][1], annual_miles)
    ratios_r = get_consistency_ratios(cohort_idx['r'], lifecycle['r'][0], lifecycle['r'][1], annual_miles)
    ratios_d = get_consistency_ratios(cohort_idx['d'], lifecycle['d'][0], lifecycle['d'][1], annual_miles)

    sns.kdeplot(ratios_c, color='#2ecc71', fill=True, label='Casual', clip=(0,1))
    sns.kdeplot(ratios_r, color='#f1c40f', fill=True, label='Recreational', clip=(0,1))
    sns.kdeplot(ratios_d, color='#e74c3c', fill=True, label='Dedicated', clip=(0,1))

    plt.xlabel("Consistency Ratio (Active Years / Career Duration)")
    plt.ylabel("Density")
    plt.title("Figure 4: Consistency of Habit by Group")
    plt.xlim(0, 1)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('output_file_consistency.png')

def plot_annual_distributions(params):
    """Generates Figure 5: Annual Volume Distributions."""
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 5000, 1000)
    
    def lognormal_pdf(x, mu, sigma):
        return (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))

    mu_c_mixed = params['mu_c_base'] + 0.5 * params['beta_gender']
    mu_r_mixed = params['mu_r_base'] + 0.5 * params['beta_gender']
    mu_d_mixed = params['mu_d_base'] + 0.5 * params['beta_gender']
    
    dists = [
        (mu_c_mixed, params['sig_c'], '#2ecc71', 'Casual Annual'),
        (mu_r_mixed, params['sig_r'], '#f1c40f', 'Rec Annual'),
        (mu_d_mixed, params['sig_d'], '#e74c3c', 'Dedicated Annual')
    ]

    for mu, sigma, color, label in dists:
        y = lognormal_pdf(x, mu, sigma)
        plt.plot(x, y, color=color, label=label, linewidth=2)
        plt.fill_between(x, y, alpha=0.3, color=color)

    plt.xlim(0, 4000)
    plt.xlabel("Annual Mileage")
    plt.ylabel("Probability Density")
    plt.title("Figure 5: Annual Volume Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig('output_file_annual_dist.png')

def plot_gantt(sim_results):
    """Generates Figure 6: Career Gantt Chart."""
    from matplotlib.lines import Line2D
    plt.figure(figsize=(12, 8))
    
    groups = sim_results['groups']
    annual_miles = sim_results['annual_miles']
    cohort_idx = sim_results['cohort_indices']
    lifecycle = sim_results['lifecycle_data']
    
    n_sample = 50
    active_indices = np.where(groups > 0)[0]
    sample_indices = np.random.choice(active_indices, min(n_sample, len(active_indices)), replace=False)

    plot_data = []
    for i in sample_indices:
        g = groups[i]
        color = {'1': '#2ecc71', '2': '#f1c40f', '3': '#e74c3c'}[str(g)]
        
        cohort_key = {1: 'c', 2: 'r', 3: 'd'}[g]
        pos = np.where(cohort_idx[cohort_key] == i)[0][0]
        start, stop = lifecycle[cohort_key][0][pos], lifecycle[cohort_key][1][pos]
        
        plot_data.append({'id': i, 'group': g, 'color': color, 'start': start, 'stop': stop, 'duration': stop-start})

    plot_data.sort(key=lambda x: (x['group'], x['duration']))

    for y_pos, p in enumerate(plot_data):
        user_miles = annual_miles[p['id']]
        for age in range(16, 81):
            if user_miles[age] > 0:
                plt.hlines(y=y_pos, xmin=age, xmax=age+1, color=p['color'], linewidth=3)

    plt.xlabel("Age")
    plt.yticks([])
    plt.ylabel("Individual Runners (Sample)")
    plt.title("Figure 6: Career Arcs (Intermittency/Gaps)")
    
    custom_lines = [Line2D([0], [0], color='#2ecc71', lw=4),
                    Line2D([0], [0], color='#f1c40f', lw=4),
                    Line2D([0], [0], color='#e74c3c', lw=4)]
    plt.legend(custom_lines, ['Casual', 'Recreational', 'Dedicated'])
    plt.xlim(16, 80)
    plt.tight_layout()
    plt.savefig('output_file_gantt.png')

def plot_scatter(sim_results, n_scatter=2000):
    """Generates Figure 7: Persistence Scatter."""
    from matplotlib.lines import Line2D
    plt.figure(figsize=(10, 6))
    
    n_sim = len(sim_results['groups'])
    scatter_idx = np.random.choice(n_sim, min(n_scatter, n_sim), replace=False)
    
    x_vals, y_vals, colors = [], [], []
    color_map = {1: '#2ecc71', 2: '#f1c40f', 3: '#e74c3c'}
    
    for i in scatter_idx:
        g = sim_results['groups'][i]
        if g == 0: continue
        
        miles = sim_results['annual_miles'][i]
        x_vals.append(np.sum(miles > 1))
        y_vals.append(np.sum(miles))
        colors.append(color_map.get(g, 'grey'))

    plt.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=15, edgecolors='none')
    plt.xlabel("Total Active Years")
    plt.ylabel("Lifetime Miles")
    plt.title("Figure 7: The Persistence Multiplier")
    
    custom_lines = [Line2D([0], [0], color=c, marker='o', linestyle='', markersize=5) for c in color_map.values()]
    plt.legend(custom_lines, ['Casual', 'Recreational', 'Dedicated'], loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_file_scatter.png')

def plot_spaghetti(sim_results, medians):
    """Generates Figure 8: Individual Lifecycles (Spaghetti Plot)."""
    plt.figure(figsize=(10, 6))
    
    groups = sim_results['groups']
    cumulative = sim_results['cumulative']
    active_indices = np.where(groups > 0)[0]
    n_spaghetti = 100
    spaghetti_idx = np.random.choice(active_indices, min(n_spaghetti, len(active_indices)), replace=False)
    
    color_map = {1: '#2ecc71', 2: '#f1c40f', 3: '#e74c3c'}
    for i in spaghetti_idx:
        plt.plot(np.arange(16, 81), cumulative[i, 16:81], color=color_map[groups[i]], alpha=0.3, linewidth=1)

    plt.plot(np.arange(16, 81), medians['c'], color='#1b5e20', linewidth=3, label='Casual Median')
    plt.plot(np.arange(16, 81), medians['r'], color='#f57f17', linewidth=3, label='Recreational Median')
    plt.plot(np.arange(16, 81), medians['d'], color='#b71c1c', linewidth=3, label='Dedicated Median')

    plt.xlabel("Age")
    plt.ylabel("Cumulative Miles")
    plt.title("Figure 8: Individual Lifecycles (Spaghetti Plot)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
    plt.tight_layout()
    plt.savefig('output_file_spaghetti.png')

def plot_diagnostics(idata):
    """Generates Figure 9 & 10: Model Diagnostics (PPC and Forest Plot)."""
    # 1. Posterior Predictive Check (Manual for better control over scalar observed data)
    obs_vars = list(idata.observed_data.data_vars)
    fig, axes = plt.subplots(len(obs_vars), 1, figsize=(10, 4 * len(obs_vars)))
    if len(obs_vars) == 1: axes = [axes]
    
    for i, var in enumerate(obs_vars):
        obs_val = idata.observed_data[var].values[0]
        pp_samples = idata.posterior_predictive[var].values.flatten()
        
        sns.kdeplot(pp_samples, ax=axes[i], label='Posterior Predictive Samples', fill=True, color='#3498db', alpha=0.4)
        axes[i].axvline(obs_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Observed Anchor ({obs_val:.2f})')
        axes[i].set_title(f"PPC Check: {var}")
        axes[i].set_xlabel("Log(Annual Miles)")
        axes[i].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output_file_diag_ppc.png')
    plt.close()
    
    # 2. Forest Plot
    plt.figure(figsize=(10, 6))
    az.plot_forest(idata, var_names=['beta_gender', 'mu_casual_base', 'mu_rec_base', 'mu_ded_base'], combined=True)
    plt.title("Figure 10: Parameter Forest Plot (94% HDI)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_file_diag_forest.png')

# ==========================================
# 4. DATA EXPORT & HTML UPDATE
# ==========================================

def get_export_data(sim_results):
    """Prepares data for export to HTML."""
    cumulative = sim_results['cumulative']
    cohort_idx = sim_results['cohort_indices']
    
    model_data_export = {}
    ages_to_export = [16, 20, 30, 40, 50, 60, 70, 80]
    
    for age in ages_to_export:
        stats = {}
        for key in ['c', 'r', 'd']:
            arr = cumulative[cohort_idx[key], age]
            active = arr[arr > 10]
            if len(active) > 0:
                stats[key] = [round(np.mean(active)), round(np.std(active))]
            else:
                stats[key] = [0, 0]
        model_data_export[age] = stats
        
    return model_data_export

def update_html_data(model_data, fig_stats, beta_gender, weights, html_path='index.html'):
    """Injects simulation results into index.html."""
    js_content = f"        const WEIGHTS = {json.dumps(weights)};\n"
    js_content += f"        const BETA_GENDER = {beta_gender:.4f};\n"
    js_content += f"        const modelData = {json.dumps(model_data, indent=12).replace('}', '        }')};\n"
    js_content += f"        const FIG_STATS = {json.dumps(fig_stats, indent=12).replace('}', '        }')};"
    
    try:
        with open(html_path, 'r') as f:
            content = f.read()
        
        pattern = r'// \[DATA_INJECTION_START\].*?// \[DATA_INJECTION_END\]'
        replacement = f'// [DATA_INJECTION_START]\n{js_content}\n        // [DATA_INJECTION_END]'
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        with open(html_path, 'w') as f:
            f.write(new_content)
        print(f"Successfully updated {html_path} with new model parameters.")
    except Exception as e:
        print(f"Error updating {html_path}: {e}")

# ==========================================
# 5. MAIN ENTRY POINT
# ==========================================

def main():
    config = Config()
    idata, params = run_mcmc(config)
    sim_results, mu_c, mu_r, mu_d = run_simulation(config, params)
    
    print("Generating Visualizations...")
    sns.set_style("whitegrid")
    
    modes = plot_pdf(config, sim_results)
    plot_cdf(sim_results)
    medians = plot_medians(sim_results)
    plot_consistency(sim_results)
    plot_annual_distributions(params)
    plot_gantt(sim_results)
    plot_scatter(sim_results)
    plot_spaghetti(sim_results, medians)
    plot_diagnostics(idata)
    
    print("Exporting data to HTML...")
    model_data = get_export_data(sim_results)
    fig_stats = {'age60': {k + 'Mode': v for k, v in modes.items()}}
    weights = {'non': config.P_NON, 'casual': config.P_CASUAL, 'rec': config.P_REC, 'ded': config.P_DEDICATED}
    beta_gender_val = float(idata.posterior['beta_gender'].mean())
    
    update_html_data(model_data, fig_stats, beta_gender_val, weights)
    print("Refactor complete. All outputs saved and index.html updated.")

if __name__ == "__main__":
    main()
