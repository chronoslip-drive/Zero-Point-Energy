import numpy as np
import matplotlib.pyplot as plt

kappa = 0.02
p = 55.89
target = 1e-120

N = np.logspace(1, 3, 300)  # N_trunc from 10 to 1000
rho = (kappa**2) / N**p

kappa_vals = np.logspace(-3, 0, 300)
rho_kappa = (kappa_vals**2) / (122**p)

kappa_vals = np.logspace(-3, 0, 300)
rho_kappa = (kappa_vals**2) / (122**p)

plt.figure(figsize=(9, 6))
plt.loglog(kappa_vals, rho_kappa, 'g-', lw=2.5, label=r'fixed $N_{\mathrm{trunc}}=122$, $p\approx55.89$')
plt.axhline(1e-120, color='red', ls='--', lw=1.5)
plt.axvline(0.02, color='gray', ls=':', lw=2, label=r'$\kappa \approx 0.02$ (toy attractor)')
plt.xlabel(r'Emergent asymmetry $\kappa$', fontsize=13)
plt.ylabel(r'$\rho_{\mathrm{vac}} / M_{\mathrm{Pl}}^4$', fontsize=13)
plt.title('Vacuum energy sensitivity to attractor value κ', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()
