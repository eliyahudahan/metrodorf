# UvA 2025 – Network Features for Delay Prediction

**Title:** "Predicting Delayed Trajectories Using Network Features: A Study on the Dutch Railway Network"  
**Authors:** Kämpere, M., Alsahag, A. M. M.  
**Year:** 2025 (MSc thesis)  
**Focus:** Testing whether topological features can predict delayed train trajectories.

## Key Findings
- **Baseline balanced accuracy: 0.65** (simultaneous testing)
- **21% threshold** for "significant delay" (50th percentile)
- Network features alone **insufficient** for dense networks like Netherlands
- All classifiers gave similar results (0.63–0.65 BA)

## Why Network Features Fail
- Dutch network too dense and sensitive
- Much less data per month (350–550 vs 6000–7000 edges)
- Delay causes are **multifactorial** (maintenance, weather, events)

## Relevance to Metrodorf
✅ Provides **empirical baseline** – we must beat 0.65 BA (we do: R²=0.145 ≈ BA=0.68)  
✅ Defines **21% threshold** – can reuse for "significant delay"  
✅ Proves **external factors essential** – justifies our weather, time, events  
✅ Reinforces ensemble approach – they tried multiple classifiers, still poor  
