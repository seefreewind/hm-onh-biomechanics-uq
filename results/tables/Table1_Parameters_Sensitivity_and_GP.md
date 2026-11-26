# Table 1. Parameter Space Specification, Empirical Copula Couplings, Global Sobol Indices, and GP Surrogate Validation

### Panel A: 10-Dimensional Ocular Biomechanical Input Space and Variance-Based Sensitivity Indices (Mean ± 95% Bootstrap CI)

| Parameter | Physical Description | Unit | LHS / Sobol Bounds | Emmetropia | High Myopia | Cohort Mean ± SD | Copula $r(AL)$ | 10D $S_T$ ($\varepsilon_{\text{eff}}$) | Intrinsic 9D $S_T$ ($\varepsilon_{\text{eff}}$) | Intrinsic 9D $S_T$ ($E_1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AL` | Axial length | mm | `[22.0, 32.0]` | `24.0` | `29.5` | `26.2 ± 2.5` | `+1.00` | `0.010 ± 0.001` | `0.034 ± 0.002` | `0.030 ± 0.002` |
| `t_ppS` | Peripapillary scleral thickness | mm | `[0.35, 1.05]` | `0.85` | `0.48` | `0.68 ± 0.16` | `-0.62` | `0.070 ± 0.005` | `0.171 ± 0.010` | `0.124 ± 0.008` |
| `theta_canal` | Neural canal obliqueness / tilt angle | deg (rad in solver) | `[0.0, 35.0]` | `3.0` | `25.0` | `12.5 ± 8.5` | `+0.58` | `0.081 ± 0.005` | `0.223 ± 0.014` | `0.420 ± 0.026` |
| `t_LC` | Lamina cribrosa central thickness | mm | `[0.12, 0.38]` | `0.26` | `0.16` | `0.23 ± 0.055` | `-0.45` | `0.031 ± 0.002` | `0.062 ± 0.004` | `0.046 ± 0.003` |
| `D_canal` | Anterior scleral canal opening diameter | mm | `[1.4, 2.4]` | `1.65` | `2.05` | `1.78 ± 0.22` | `+0.42` | `0.018 ± 0.001` | `0.020 ± 0.001` | `0.017 ± 0.001` |
| `c1_sclera` | Scleral ground matrix Mooney-Rivlin stiffness c1 | MPa | `[0.08, 0.45]` | `0.22` | `0.14` | `0.21 ± 0.075` | `-0.30` | `0.023 ± 0.002` | `0.066 ± 0.005` | `0.116 ± 0.008` |
| `ksi_fiber` | Peripapillary collagen fiber stiffness modulus ksi | MPa | `[0.8, 6.0]` | `2.8` | `2.0` | `2.75 ± 1.05` | `-0.20` | `0.126 ± 0.008` | `0.350 ± 0.019` | `0.211 ± 0.012` |
| `b_fiber` | Peripapillary circumferential fiber Von Mises concentration | dimensionless | `[0.4, 4.5]` | `2.5` | `1.1` | `2.2 ± 0.85` | `-0.48` | `0.021 ± 0.001` | `0.048 ± 0.003` | `0.028 ± 0.002` |
| `E_LC` | Lamina cribrosa effective Young's modulus | MPa | `[0.1, 0.6]` | `0.3` | `0.24` | `0.3 ± 0.1` | `-0.18` | `0.054 ± 0.004` | `0.105 ± 0.006` | `0.102 ± 0.006` |
| `IOP` | Intraocular pressure | mmHg (×0.000133322 MPa) | `[10.0, 40.0]` | `15.0` | `15.0` | `16.0 ± 3.5` | `+0.12` | `0.630 ± 0.031` | `Fixed (21 mmHg)` | `Fixed (21 mmHg)` |

### Panel B: Matérn-5/2 ARD Gaussian Process Surrogate Performance on Unseen Test Set ($N_{\text{train}}=400, N_{\text{test}}=80$)

| Quantity of Interest (QoI) | Aggregation | 5-Fold Train CV $Q^2$ | Unseen Test $Q^2$ | Test NRMSE (%) | Test RMSE | Test MAE | Log-Space GP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LC 1st Principal Tensile Strain $E_1$ (`lc_e1_p95`) | 95th %ile (Vol.-Weighted) | `0.9948` | **`0.9952`** | **`1.51%`** | `0.0696%` | `0.0507%` | Yes |
| LC Effective Strain $\varepsilon_{\text{eff}}$ (`lc_eeff_p95`) | 95th %ile (Vol.-Weighted) | `0.9962` | **`0.9968`** | **`1.27%`** | `0.1238%` | `0.0909%` | Yes |
| LC Maximum Shear Strain $\gamma_{\max}$ (`lc_gmax_p95`) | 95th %ile (Vol.-Weighted) | `0.9951` | **`0.9962`** | **`1.38%`** | `0.0719%` | `0.0522%` | Yes |
| LC Posterior Displacement vs BMO (`lc_ap_disp_p95`) | 95th %ile (Vol.-Weighted) | `0.9948` | **`0.9965`** | **`1.54%`** | `0.631 μm` | `0.439 μm` | No |
