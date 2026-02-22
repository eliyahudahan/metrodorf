# Modeling High-Speed Rail for Germany – Gravity Model for Polycentric Regions

## The Gravity Model
The core formula for ridership prediction:

\[
\mbox{Ridership} = \mbox{Pop}_{A}^{0.8} \cdot \mbox{Pop}_{B}^{0.8} / \mbox{distance}^{2}
\]

With time adjustment:
\[
\mbox{Ridership} = 1.8 \cdot \mbox{Pop}_{A}^{0.8} \cdot \mbox{Pop}_{B}^{0.8} / \max\{2.5, \mbox{time}\}^{2}
\]

## Key Insights for Polycentric Regions

### Rhine-Ruhr Region
- **Population:** 17.6 million (larger than Paris and London)
- **Structure:** Multiple interconnected cities (Dortmund, Essen, Cologne, Düsseldorf, etc.)

### Other Polycentric Regions Mentioned
- **Randstad (Netherlands):** Utrecht (1M), Amsterdam (2.5M), Rotterdam (3.5M) – acts like a monocentric region of 9 million
- **Frankfurt-Mannheim region:** Frankfurt (3.7M), Mainz (0.6M), Rhine-Neckar (2.4M), Karlsruhe (1.1M), Stuttgart (2.5M)

## Key Finding for Metrodorf
The model predicts that **Rhine-Ruhr and Frankfurt-Mannheim regions combined generate 25 million annual trips**. This confirms the region's importance and justifies our focus on polycentric delay prediction.

## Why This Matters
Traditional models treat these cities as **single points**. Levy shows they function as **interconnected megaregions** – exactly the problem we're solving with Metrodorf.
