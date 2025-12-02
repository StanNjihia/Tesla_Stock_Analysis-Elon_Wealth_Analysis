import numpy as np
import pandas as pd
import json
from scipy import stats
import pickle

# =============================================================================
# ENHANCED MONTE CARLO WITH EXTENDED PROJECTIONS
# This script runs simulations and tracks when EVERY path reaches $1T
# =============================================================================

print("=" * 70)
print("ENHANCED MONTE CARLO SIMULATION")
print("Tracking when ALL paths reach $1 Trillion")
print("=" * 70)

# Configuration
N_SIM = 10000
INITIAL_MONTHS = 60  # 5 years (our forecast window)
MAX_EXTENDED_MONTHS = 180  # Extend up to 15 years total if needed
TARGET_WEALTH = 1000  # $1 Trillion

# Current state (adjust these to your actual values)
CURRENT_TSLA_PRICE = 350.0
CURRENT_SHARES_MILLIONS = 411
CURRENT_OWNERSHIP_PCT = 12.8
CURRENT_OTHER_WEALTH = 200  # Billions
CURRENT_NETWORTH = (CURRENT_TSLA_PRICE * CURRENT_SHARES_MILLIONS / 1000) + CURRENT_OTHER_WEALTH

# Historical parameters (calibrated from your data)
TSLA_MONTHLY_RETURN_MEAN = 0.015  # ~20% annual
TSLA_MONTHLY_RETURN_STD = 0.12    # ~42% annual volatility
OTHER_WEALTH_ANNUAL_GROWTH = 0.20  # 20% annual
OTHER_WEALTH_ANNUAL_STD = 0.10     # 10% volatility

print(f"\nSimulation Parameters:")
print(f"  Starting Net Worth: ${CURRENT_NETWORTH:.1f}B")
print(f"  TSLA Price: ${CURRENT_TSLA_PRICE:.2f}")
print(f"  Shares Owned: {CURRENT_SHARES_MILLIONS}M")
print(f"  Other Wealth: ${CURRENT_OTHER_WEALTH:.1f}B")
print(f"  Simulations: {N_SIM:,}")
print(f"  Initial Window: {INITIAL_MONTHS} months (5 years)")
print(f"  Max Extension: {MAX_EXTENDED_MONTHS} months (15 years total)")

# =============================================================================
# SIMULATION FUNCTION
# =============================================================================

def simulate_single_path(sim_id, extend_to_trillion=True):
    """
    Simulate a single wealth path.
    If extend_to_trillion=True, continue beyond 5 years until $1T is reached.
    """
    np.random.seed(sim_id)  # Reproducible results
    
    # Initialize
    tsla_price = CURRENT_TSLA_PRICE
    other_wealth = CURRENT_OTHER_WEALTH
    shares = CURRENT_SHARES_MILLIONS
    
    # Storage
    path_data = {
        'months': [],
        'tsla_price': [],
        'tsla_wealth': [],
        'other_wealth': [],
        'total_wealth': [],
        'reached_1t': False,
        'month_reached_1t': None
    }
    
    month = 0
    reached_target = False
    
    # Convert annual rates to monthly
    monthly_other_growth = (1 + OTHER_WEALTH_ANNUAL_GROWTH) ** (1/12) - 1
    monthly_other_std = OTHER_WEALTH_ANNUAL_STD / np.sqrt(12)
    
    # Simulate up to MAX_EXTENDED_MONTHS or until $1T is reached
    while month <= MAX_EXTENDED_MONTHS:
        # TSLA monthly return
        tsla_return = np.random.normal(TSLA_MONTHLY_RETURN_MEAN, TSLA_MONTHLY_RETURN_STD)
        tsla_price *= (1 + tsla_return)
        tsla_price = max(tsla_price, 1)  # Floor at $1
        
        # TSLA wealth
        tsla_wealth = (tsla_price * shares) / 1000  # In billions
        
        # Other wealth growth
        other_return = np.random.normal(monthly_other_growth, monthly_other_std)
        other_wealth *= (1 + other_return)
        other_wealth = max(other_wealth, 0)
        
        # Total wealth
        total_wealth = tsla_wealth + other_wealth
        
        # Store data
        path_data['months'].append(month)
        path_data['tsla_price'].append(tsla_price)
        path_data['tsla_wealth'].append(tsla_wealth)
        path_data['other_wealth'].append(other_wealth)
        path_data['total_wealth'].append(total_wealth)
        
        # Check if reached $1T
        if not reached_target and total_wealth >= TARGET_WEALTH:
            reached_target = True
            path_data['reached_1t'] = True
            path_data['month_reached_1t'] = month
            
            # If we've passed initial window and reached target, we can stop
            if month >= INITIAL_MONTHS or not extend_to_trillion:
                break
        
        # If we're past initial window and haven't reached $1T yet
        if month >= INITIAL_MONTHS and not extend_to_trillion:
            break
        
        # If we reached target before initial window, continue to initial window
        if reached_target and month < INITIAL_MONTHS:
            month += 1
            continue
        
        # If we reached target after initial window, we can stop
        if reached_target and month >= INITIAL_MONTHS:
            break
            
        month += 1
    
    return path_data

# =============================================================================
# RUN ALL SIMULATIONS
# =============================================================================

print("\nRunning simulations...")
print("This may take 2-5 minutes for 10,000 simulations...")

all_paths = []
paths_reaching_1t_in_window = 0
paths_reaching_1t_extended = 0
paths_never_reaching_1t = 0

for sim in range(N_SIM):
    path = simulate_single_path(sim, extend_to_trillion=True)
    all_paths.append(path)
    
    # Count outcomes
    if path['reached_1t']:
        if path['month_reached_1t'] <= INITIAL_MONTHS:
            paths_reaching_1t_in_window += 1
        else:
            paths_reaching_1t_extended += 1
    else:
        paths_never_reaching_1t += 1
    
    # Progress indicator
    if (sim + 1) % 1000 == 0:
        print(f"  Completed {sim + 1:,}/{N_SIM:,} simulations...")

print("\n" + "=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)

# Calculate probabilities
prob_in_window = (paths_reaching_1t_in_window / N_SIM) * 100
prob_extended = (paths_reaching_1t_extended / N_SIM) * 100
prob_never = (paths_never_reaching_1t / N_SIM) * 100
prob_eventually = prob_in_window + prob_extended

print(f"\n✅ Within 5 years (by Nov 2030): {paths_reaching_1t_in_window:,} paths ({prob_in_window:.1f}%)")
print(f"⏰ After 5 years (6-15 years):    {paths_reaching_1t_extended:,} paths ({prob_extended:.1f}%)")
print(f"❌ Never reaches $1T (15+ years): {paths_never_reaching_1t:,} paths ({prob_never:.1f}%)")
print(f"\n🎯 EVENTUALLY reaches $1T:       {paths_reaching_1t_in_window + paths_reaching_1t_extended:,} paths ({prob_eventually:.1f}%)")

# =============================================================================
# ANALYZE TIME-TO-TRILLION
# =============================================================================

months_to_trillion = [p['month_reached_1t'] for p in all_paths if p['reached_1t']]

if len(months_to_trillion) > 0:
    print("\n" + "=" * 70)
    print("TIME TO $1 TRILLION ANALYSIS")
    print("=" * 70)
    
    print(f"\nFastest path:  {min(months_to_trillion)} months ({min(months_to_trillion)/12:.1f} years)")
    print(f"Median path:   {int(np.median(months_to_trillion))} months ({np.median(months_to_trillion)/12:.1f} years)")
    print(f"Slowest path:  {max(months_to_trillion)} months ({max(months_to_trillion)/12:.1f} years)")
    
    # Time brackets
    within_5yr = sum(1 for m in months_to_trillion if m <= 60)
    within_7yr = sum(1 for m in months_to_trillion if m <= 84)
    within_10yr = sum(1 for m in months_to_trillion if m <= 120)
    
    print(f"\nCumulative Probability:")
    print(f"  Within 5 years:  {(within_5yr/N_SIM)*100:.1f}%")
    print(f"  Within 7 years:  {(within_7yr/N_SIM)*100:.1f}%")
    print(f"  Within 10 years: {(within_10yr/N_SIM)*100:.1f}%")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# 1. Save complete paths (pickle for efficiency)
with open('data/monte_carlo_all_paths.pkl', 'wb') as f:
    pickle.dump(all_paths, f)
print("✅ Saved all simulation paths: data/monte_carlo_all_paths.pkl")

# 2. Save summary statistics
summary = {
    'n_simulations': N_SIM,
    'target_wealth': TARGET_WEALTH,
    'initial_months': INITIAL_MONTHS,
    'paths_in_window': paths_reaching_1t_in_window,
    'paths_extended': paths_reaching_1t_extended,
    'paths_never': paths_never_reaching_1t,
    'prob_in_window': prob_in_window,
    'prob_extended': prob_extended,
    'prob_eventually': prob_eventually,
    'fastest_months': min(months_to_trillion) if months_to_trillion else None,
    'median_months': int(np.median(months_to_trillion)) if months_to_trillion else None,
    'slowest_months': max(months_to_trillion) if months_to_trillion else None,
    'current_networth': CURRENT_NETWORTH,
    'current_tsla_price': CURRENT_TSLA_PRICE,
    'current_shares_millions': CURRENT_SHARES_MILLIONS,
    'current_other_wealth': CURRENT_OTHER_WEALTH
}

with open('data/monte_carlo_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✅ Saved summary statistics: data/monte_carlo_summary.json")

# 3. Save sample paths for visualization (100 random paths)
np.random.seed(42)
sample_indices = np.random.choice(N_SIM, size=min(100, N_SIM), replace=False)
sample_paths = [all_paths[i] for i in sample_indices]

with open('data/monte_carlo_sample_paths.pkl', 'wb') as f:
    pickle.dump(sample_paths, f)
print("✅ Saved 100 sample paths: data/monte_carlo_sample_paths.pkl")

# 4. Create percentiles dataframe (for existing visualization)
percentiles_data = []
for month in range(INITIAL_MONTHS + 1):
    month_values = []
    for path in all_paths:
        if month < len(path['total_wealth']):
            month_values.append(path['total_wealth'][month])
    
    if month_values:
        percentiles_data.append({
            'Month': month,
            'P5': np.percentile(month_values, 5),
            'P10': np.percentile(month_values, 10),
            'P25': np.percentile(month_values, 25),
            'Median': np.percentile(month_values, 50),
            'P75': np.percentile(month_values, 75),
            'P90': np.percentile(month_values, 90),
            'P95': np.percentile(month_values, 95),
            'Mean': np.mean(month_values)
        })

percentiles_df = pd.DataFrame(percentiles_data)
# Create date index
from datetime import datetime, timedelta
start_date = datetime(2025, 11, 30)
dates = [start_date + timedelta(days=30*i) for i in range(len(percentiles_df))]
percentiles_df.index = dates

percentiles_df.to_csv('data/monte_carlo_percentiles.csv')
print("✅ Saved percentiles: data/monte_carlo_percentiles.csv")

# 5. Save time-to-trillion distribution
time_dist = pd.DataFrame({
    'simulation_id': range(len(all_paths)),
    'reached_1t': [p['reached_1t'] for p in all_paths],
    'months_to_1t': [p['month_reached_1t'] if p['reached_1t'] else None for p in all_paths],
    'final_wealth_5yr': [p['total_wealth'][min(60, len(p['total_wealth'])-1)] for p in all_paths]
})

time_dist.to_csv('data/monte_carlo_time_to_trillion.csv', index=False)
print("✅ Saved time-to-trillion data: data/monte_carlo_time_to_trillion.csv")

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
print("\nYou can now use these files in your Streamlit app to:")
print("  • Show interactive simulation paths")
print("  • Display when 'failed' paths eventually reach $1T")
print("  • Let users explore individual scenarios")
print("  • Show time-to-trillion distributions")
print("\nNext: Update your Streamlit app to use these enhanced results!") 
