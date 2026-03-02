# Bologna 2025 – Power Laws in Railway Delays

**Title:** "Emergence of power laws in hierarchical dynamics on multi‑level graphs"  
**Authors:** Rondini, T., Berselli, G., Degli Esposti, M., Bazzani, A.  
**Preprint:** arXiv:2509.18782v1 (23 Sep 2025)  
**Focus:** How simple priority rules generate heavy‑tailed delay distributions.

## Key Empirical Findings
1. **Station-to-station delays** fit an Asymmetric Laplace distribution
2. **Whole-trip delays:**
   - German high-speed: pure exponential
   - Italian high-speed: exponential + power-law tail (cut-off at 30 min)
   - Local trains: pure power-law (heavy tails)
3. **Priority rules** explain cut-offs – trains downgraded after 30/60 min delay

## Model & Simulation
- Queue-based model on station network
- Management time τ = 5 minutes
- Added Laplacian noise based on empirical distributions
- Results reproduce observed patterns

## Relevance to Metrodorf
✅ Provides mathematical foundation for using **Gaussian (Laplacian) models** – they capture heavy tails  
✅ Explains why **Gaussian model gets 47.4% weight** (local trains = heavy tails)  
✅ Priority rules justify **Cologne bottleneck effect**  
