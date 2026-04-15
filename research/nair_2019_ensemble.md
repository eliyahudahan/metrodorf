# Nair et al. (2019) - Ensemble Prediction for Train Delays

**Journal:** Transportation Research Part C: Emerging Technologies
**DOI:** 10.1016/j.trc.2019.04.026

## Key Findings:
- Ensemble of random forest + kernel regression + simulation
- 25% improvement over schedule
- Used Deutsche Bahn data (25,000 trains/day)

## Relevance to Metrodorf:
- Confirms ensemble approach (Al Ghamdi 2022)
- Context-aware weights (time of day, train class)
- Weather and events matter

## Metrics:
- % within 1 minute
- RMSE (seconds)

## Action items:
- [ ] Compare our MAE (2.75 min) to their RMSE
- [ ] Consider adding weather API
- [ ] Consider dynamic weights by time-of-day
