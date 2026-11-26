# Dynamic Structural Econometrics — Teaching Materials

Coding workbooks on estimating dynamic structural models. Code workbooks are available in MATLAB for all topics. Code in Julia for all topics is also provided. You can also find R code for the base CCP estimation. Materials are also published as standalone HTML via GitHub Pages.

- **Instructor:** Aaron Barkley — <https://sites.google.com/site/aaronmbarkley>
- **Source code & repository:** <https://github.com/ambarkley/Dynamic-Structural-Models>
- **Rendered site (GitHub Pages):** <https://ambarkley.github.io/Dynamic-Structural-Models/>

---

## Workbooks

- **Bus Engine Replacement — CCP Estimation**  
  Rust (1987) bus engine replacement as the workhorse DDC example.  
  Conditional Choice Probability (CCP) estimator per Hotz & Miller (1993).  
  Arcidiacono & Ellickson (2011) formulation and implementation details.  
  Full-information MLE (à la Rust) included for comparison.

- **Unobserved Heterogeneity — EM Algorithm**  
  Step-by-step EM algorithm with a simple integer-data illustration.  
  Application to DDC models following Arcidiacono & Miller (2011).  
  

- **Dynamic Games — Entry/Exit**  
  From single-agent DDC to multi-agent, simultaneous-move settings.  
  Entry/exit game as the empirical IO benchmark example.  
  Adapting single-agent estimators to fixed-point equilibria in games.


- **Continuous Choice — Consumption/Saving**  
  Hansen & Singleton (1982) Euler-equation framework.  
  Estimating utility parameters using aggregate consumption and asset returns.  
  

- **Auctions — Static & Dynamic**  
  Estimation for first-price and second-price sealed-bid auctions.  
  Data-limited settings (e.g., only the winning bid) and identification strategies.  
  Extensions to dynamic ascending (English) auctions under frictionless bidding.  


---

## Julia code

Julia code is also available for all topics above. The Manifest and Project files are included in the repo -- activate/instantiate to use the required packages. Some of the code may slightly differ from the MATLAB versions.

