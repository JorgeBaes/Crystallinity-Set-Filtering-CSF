import numpy as np
import pandas as pd

# ============================================================
# Crystallinity-Set Filtering (CSF) Method Implementation
# ============================================================
# This script estimates the mass fraction of HDPE and PP in
# semicrystalline polymer blends using DSC melting enthalpy data.
# The method explores a discretized crystallinity set and applies
# crystallinity constraints to filter valid solutions.
# ============================================================


# ------------------------------------------------------------
# Reference melting enthalpies for 100% crystalline polymers (J/g)
# ------------------------------------------------------------
H0_HDPE = 293
H0_PP = 170


# ------------------------------------------------------------
# Crystallinity bounds (%)
# These represent typical crystallinity ranges for each polymer
# ------------------------------------------------------------
cryst_HDPE = {'min': 40, 'max': 90}
cryst_PP = {'min': 30, 'max': 60}


# ------------------------------------------------------------
# Method parameters
# epsilon: discretization step of crystallinity set
# delta: tolerance for mass balance filtering
# ------------------------------------------------------------
epsilon = 0.1
delta = 0.001


# ------------------------------------------------------------
# Experimental data (DSC)
# Format:
# 'composition': (ΔH_HDPE, ΔH_PP, nominal_HDPE_fraction)
# ------------------------------------------------------------
blends = {
    '5/95':  (8.73, 77.45, 0.05),
    '10/90': (17.17, 73.48, 0.10),
    '20/80': (41.26, 61.38, 0.20),
    '40/60': (78.73, 44.07, 0.40),
    '50/50': (85.27, 41.17, 0.50),
    '60/40': (112.85, 29.92, 0.60),
    '80/20': (144.70, 16.56, 0.80),
    '90/10': (151.30, 7.22, 0.90),
    '95/5':  (171.90, 2.56, 0.95),
}


# ------------------------------------------------------------
# Storage for results
# ------------------------------------------------------------
results = []
nominal_values = []
estimated_values = []


# ============================================================
# Main loop: process each blend composition
# ============================================================
for label, (H_HDPE, H_PP, nominal_HDPE) in blends.items():

    crystallinity_set = []

    # --------------------------------------------------------
    # Build crystallinity set C
    # --------------------------------------------------------
    x = cryst_HDPE["min"]
    while x <= cryst_HDPE["max"]:
        y = cryst_PP["min"]
        while y <= cryst_PP["max"]:

            # Physical constraint: HDPE crystallinity > PP crystallinity
            if y < x:

                # Calculate mass fractions based on crystallinity pair (x, y)
                phi_HDPE = H_HDPE / (H0_HDPE * x / 100)
                phi_PP = H_PP / (H0_PP * y / 100)

                crystallinity_set.append({
                    'x_HDPE': x,
                    'y_PP': y,
                    'phi_HDPE': phi_HDPE,
                    'phi_PP': phi_PP,
                    'phi_total': phi_HDPE + phi_PP
                })

            y += epsilon
        x += epsilon

    # --------------------------------------------------------
    # Apply filtering condition: mass sum ~ 1
    # --------------------------------------------------------
    filtered_set = [
        el for el in crystallinity_set
        if (1 - delta <= el['phi_total'] <= 1 + delta)
    ]

    # Extract values
    phi_HDPE_values = [el['phi_HDPE'] for el in filtered_set]
    phi_PP_values = [el['phi_PP'] for el in filtered_set]

    # --------------------------------------------------------
    # Statistical analysis
    # --------------------------------------------------------
    mean_HDPE = np.mean(phi_HDPE_values)
    mean_PP = np.mean(phi_PP_values)

    std_HDPE = np.std(phi_HDPE_values, ddof=1)
    std_PP = np.std(phi_PP_values, ddof=1)

    abs_error_HDPE = abs(mean_HDPE - nominal_HDPE)
    abs_error_PP = abs(mean_PP - (1 - nominal_HDPE))

    # Store for correlation analysis
    nominal_values.append(nominal_HDPE)
    estimated_values.append(mean_HDPE)

    # --------------------------------------------------------
    # Store results (formatted for output)
    # --------------------------------------------------------
    results.append({
        'Composition': label,
        'N': len(filtered_set),
        'HDPE (%)': str(round(mean_HDPE * 100, 2)),
        'HDPE Min': str(round(min(phi_HDPE_values) * 100, 2)),
        'HDPE Max': str(round(max(phi_HDPE_values) * 100, 2)),
        'HDPE Error': str(round(abs_error_HDPE * 100, 2)),
        'HDPE Std': str(round(std_HDPE * 100, 2)),

        'PP (%)': str(round(mean_PP * 100, 2)),
        'PP Min': str(round(min(phi_PP_values) * 100, 2)),
        'PP Max': str(round(max(phi_PP_values) * 100, 2)),
        'PP Error': str(round(abs_error_PP * 100, 2)),
        'PP Std': str(round(std_PP * 100, 2)),
    })


# ============================================================
# DataFrame for visualization
# ============================================================
df_results = pd.DataFrame(results)


# ============================================================
# Model evaluation
# ============================================================
pearson_corr = np.corrcoef(nominal_values, estimated_values)[0][1]
slope, intercept = np.polyfit(nominal_values, estimated_values, 1)


# ============================================================
# Console output
# ============================================================
print("Blend estimation method (CSF)")
print("Epsilon:", epsilon)
print("Delta:", delta)
print("Reference enthalpies (HDPE | PP):", H0_HDPE, H0_PP)
print("Crystallinity bounds (HDPE | PP):", cryst_HDPE, cryst_PP)
print("\nResults:\n")

print("Pearson correlation coefficient:", round(pearson_corr,4))
print("Linear regression:")
print("Nominal = (", round(slope,4), ") x Estimated + (", round(intercept,4), ")")

# Table output
print("\nComposition\tN\tHDPE\tMin\tMax\tErr\tStd\tPP\tMin\tMax\tErr\tStd")
for _, row in df_results.iterrows():
    print(
        f"{row['Composition']}\t{row['N']}\t{row['HDPE (%)']}\t{row['HDPE Min']}\t{row['HDPE Max']}\t"
        f"{row['HDPE Error']}\t{row['HDPE Std']}\t{row['PP (%)']}\t{row['PP Min']}\t"
        f"{row['PP Max']}\t{row['PP Error']}\t{row['PP Std']}"
    )
