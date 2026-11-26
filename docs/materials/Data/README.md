# Data for the Continuous Choices teaching illustration

The MATLAB and Julia files use the same four public-domain FRED series:

- `A794RX0Q048SBEA`: real personal consumption expenditures per capita, quarterly (BEA).
- `DTB3`: 3-month Treasury bill secondary-market rate, daily (Federal Reserve Board).
- `DGS10`: 10-year Treasury constant-maturity yield, daily (Federal Reserve Board).
- `CPIAUCSL`: consumer price index, monthly (BLS).

The code restricts the common sample to 1962Q1-2019Q4. It constructs a
three-month bill return from `DTB3` and a simplified 10-year zero-coupon holding
return from `DGS10`. These choices are for teaching and are not a replication of
Hansen and Singleton (1982).

Sources: [FRED series A794RX0Q048SBEA](https://fred.stlouisfed.org/series/A794RX0Q048SBEA),
[DTB3](https://fred.stlouisfed.org/series/DTB3),
[DGS10](https://fred.stlouisfed.org/series/DGS10), and
[CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL). Citation is requested by
the source agencies. The files are snapshots; set `readLocal = false` to retrieve
current vintages from FRED.

No WRDS/CRSP data are included because access to WRDS does not imply permission
to redistribute vendor data publicly. The prior NASDAQ file is also omitted
because that FRED series is copyrighted by Nasdaq.

## Bus Engine CCP teaching illustration

`BusEngineCCP_data.csv` is simulated data, not data from Rust (1987). It contains
1,000 buses observed for 15 periods under the model used in the MATLAB, Julia,
and R examples. The columns are bus identifier, period, mileage-state index,
mileage, bus type, and decision. Decision 1 is replace and decision 2 is
continue. The simulated file may be distributed with the teaching materials.
