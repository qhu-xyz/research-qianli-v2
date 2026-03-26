# porting
## pjm and miso annual trade loading -> miso will use this same functionality. This is cannonical
trades = aptools.get_trades_of_given_duration(participant=None, start_month='2018-06', end_month_in='2026-06')
trades_annual = trades[trades['period_type'].isin(aptools.tools.annual_period_types)].copy()
trades_annual = aptools.tools.break_offpeak(trades_annual) # my teammate told me this is not needed for miso and does not affect miso, you need to check.
trades_annual = trades_annual[trades_annual['class_type'].isin(aptools.tools.classtypes)].copy()
--> for backtesting, some of the functions above take long, better cache everything. 
But we need to **verify the results you've been getting, the paths are consistent with our previous findings**
- also you need to verify for miso, then pjm, for which years we have data

## planning of archiving and porting over to new data sources
**We are now using new data sources**
- miso first
- put everything we've done under archive. Especially the documents so as NOT to confuse ourselves. 
- archive
    - miso
    - pjm
and start miso afresh. 
- load new data sources, then verify if the number of rows and rounds and years we've been getting at this new data source is consistent with our old data. 
- then there is the baseline for miso. using this new data to test the older models under registry
    - then verify: does adding a bit of 1(rev) - defined below, add information? check /pbase or skills to see how to load 1(rev)
    Notice: in 2026 auction, happening around early 2026, april, which is for 2027 py annual, we CANNOT use information after 03-31. thus for quarters, you need to be very careful with data are not leaky. for example: if we are predicting miso's r, aq1's mcp, we were using nodal stitch; but you need to check if 2025-06, 07, 08's realized DA are helpful? and you can define 1(rev) as sum. but for aq4, covering mar, apr, may, we can only use 2026-03's data (or previous)
- then for baseline we need to consider:
    - flow type
    - 1(rev) as bins, currently were we using baseline as bins? or some combination? 


## status for now before porting
Baseline for both miso and pjm mostly fixed
- miso uses nodal stich for r1, pjm uses long-term round 5

## now for banding and baseline (please see planning of archiving and porting over to new data sources)
- if we use flow type: then what is the basic unit for such a quantile-definition? are you grouping by:
(year, round, bin, flow type, class_type?) if we do so, what is the statistics for such a groupby? gimme min and pct10
- **we should also consider 1(rev) as a direct feature/grouping element**
    - one thing we've noticed: the more 1(rev) grows/shrinks, the more this path is contended and true mcp will be higher; thus if we do not try to divide by 1(rev) as another bucket divider, we might lose valuable information.
    - but then for PJM, what is the 1(rev)?
    Several things to notice: 
    1. the pjm/miso auction starts around april-07 per year and data after this cutoff should be considered leak
    2. start of each planning year is june-01.
    3. we can sum this period to get 1(rev) for this previous year, then sum over the previous previous year's rev to get 2(rev)
    4. or maybe the recent months's revenue is the most relevant
- and that got me thinking (let's check this first):
    - we used mtm_1st_mean which is long-term round5's mcp as round1 baseline. do you think we can also develop a better mcp prediction similar to f0p's blending? 
    - find each path's march's revenue and see if it helps in predicting round1.
   

## now write me a porting plan, first focusing on miso. gimme very detailed, verifiable steps for independent audit.

1. have you used normalization for both f0 and 1(rev) when doing baseline?
2. for q5, what formula is used for baseline now? any simple way you can improve?
3. do we have any normalization applied for banding? width p95 (or at any level to 12k is not acceptable). previously we used 15k as the absolute cap and -15k as floor

*** in our repo, or later in the params config we might need to devote a separate section to make everything a bit normalized/scaling/hardcap for better stability


1. are you separating ctypes?
2. again, gimme how you've defined 1(rev) and f0 nodal for each period type. any nans? are you making any silent fallbacks that you shouldn't be using?
3. our bins are divided by baseline? what if we use 1(rev)? i think the bands might depend heavily on


## add in more features
- signals for baseline calculation: check /home/xyz/workspace/research-qianli-v2/research-annual-signal and see how to consume them
- we've only added 1(rev) for the corresponding period - but does the overall performance of previous year help?



## merge what we have into production
1. pull pmodel dev first - locate band_generator.py
2. study how ftr24/v2 calls this function in monthly models. 

### our purpose:
1. if we pass in a trades of any year, regardless of how many rows it have, it can produce 10 bid prices with 10 bid cps.
- does the current v2's f0p band_generator allow this? i think it does.
- does our plan allow this?

### Important detail:
current in research repo the bid price points in 3 month (miso) and 12 month (pjm)
- for production, we need to divide into monthly prices. 
*****
other things we need to decide:
- as you can see in f0p, there are many params you can pass into the geneating 10 bid prices/clearing probabilities
    - we need to let user decide and pass in 10 bid prices for prevail, 10 for counter. Unlike f0p, we name them as clear_5, clear_95...  (5 pct clearing probabilities for example). the clearing probs are easy but different, right? because we chose the bid ends in this repo directly calibrated by previous clearing prob. 
    - there is no ML model for baseline, ignore that
    - and there is no learned weight for the baseline also
    - if there are other important details or distinctions or blockers I've missed, raise it
    - **goal is to plug into production, so whatever concerns or gaps you have, raise it if you don't think it is evident**
- look up sell logic to see how sell prices and clear probs should be handled.

3. auction_type == "a" => that is ur cue to use our annual band_generator
4. conform to the current input/output. Stick to the params format whenever you can, but the end goal is to use band_generator as the interface to generate 10 monthly prices with clear probs. Whenever the settings have to differ, differ.
Again, for banding: we do not use banding, and our clearing probabilities come before our prices. 
5. do not generate duplicate columns **check this**
6. where to put params => /home/xyz/workspace/pmodel/src/pmodel/base/ftr24/v2/config/versions/pjm_a_onpeak/param1.json. **Make sure there is NO default in any other places besides the json file. Do NOT use any fallback unless it is absolutely necessary and right to do so, do not do it to NOT break the code** Do not default to onpeak, or whatever 
- "No fallback messages — all cells have ≥200 rows" if a fallback is to ensure data quality or integrity, always RAISE rather than fallback. No try catch. 


questions:
1. in ftr24/v2 we have this toggle to save right? i think for annual pjm and miso we need a similar same toggle. 
- when switched to "on", we save to the location. 
- notice this: you said "I need to save the recent_1 feature as a parquet so the frozen script can load it" so does it mean that it is better if we save some features (but besides 1(rev) i can't think of anything). maybe you should attach each path's 1(rev) before hand?

- document this process well
- verify end-to-end. write ample test cases.


1. current v2 annual approach is a placeholder. we need to plugin our logic. 
2. i don't know about "do we conform to v2's monthly-first approach, or override?" i dont know what this means. but we need monthly prices, prices in monthly scale.
3. yes use clear_n
4. "v2 baseline is wrong for annual" of course. v2 annual is placeholder
5. configs: i seems to remember other places need config, such as miso's baseline, which is divided into bins?
6. "For annual, we need recent_1 (PJM) or 1(rev) (MISO) pre-attached to the
  training data before band generation runs." agree --> so we do need to save toggle right? but there is one question: the f0p pipeline saves data going thru some path selection or stuff like this --> we do not need that right? so the places where the training trades are saved will be quite different?

  # other concerns
  - save clear_n n at 5, 10, 15, ...., until 95 => this is for user to choose ten from?
  - we have strict capping/flooring of 15k, 12k for baseline
  - are we caching all necessary columns?


- so the 19 bid price point values will be produced regardless whether user chooses them       
  right? because if not, i can't see how the scaling would work. (or we can just produce the bid 
   prices and cap for the most extreme ones and then scale? ==> maybe this is better?) 
- 'Reuses V2 save pattern' what does this mean? we are not going thru the path selection in annual right?


## other refactoring
- it is more prefreable to do annual banding at monthly scale naitively - in research we've done the opposite. now in annual, since prices are monthly, we should change.


## canonical load data logic
## this contains all the data loading you need; no need to do nodal replacement anymore
## this contains mtm and mcp loading logic
---
  MCP Data with Backward Fill

  Use the load_data_with_replacement or load_monthly_data_with_replacement methods on any MCP loader (PJM, MISO, etc.).

  Single auction month

  from pbase.data.dataset.ftr.mcp.pjm import PjmMcp

  mcp = PjmMcp()
  df = mcp.load_data_with_replacement(
      auction_month="2026-01-01",
      market_round=1,
      period_type="f0",
  )

  Date range

  df = mcp.load_monthly_data_with_replacement(
      from_month="2025-06-01",
      to_month="2026-01-01",
      market_round=1,
      period_type="f0",
  )

  For MISO, use MisoMcp from pbase.data.dataset.ftr.mcp.miso.

  Under the hood, these methods:
  1. Load raw MCP data via load_data()
  2. Load the nodal replacement table via self._replacement_loader.load_data()
  3. Group by class_type and call prepare_backward_fill() with node_col="pnode_id", time_col="auction_date", value_col="mcp"

  ---
  DA Data with Backward Fill

  Use load_data_with_replacement on DaLmpDailyAgg or DaLmpMonthlyAgg (in pbase.data.dataset.da.base). These are abstract — use
   the RTO-specific subclass.
  DA Daily Agg — single month
  from pbase.data.dataset.da.pjm import PjmDaLmpDailyAgg  # or equivalent RTO subclass
  da_daily = PjmDaLmpDailyAgg()
  df = da_daily.load_data_with_replacement(loading_month="2026-01-01")
  DA Daily Agg — date range
  df = da_daily.load_data_with_replacement(
      from_date="2025-06-01",
      to_date="2026-01-01",
  )
  DA Monthly Agg — single month
  from pbase.data.dataset.da.pjm import PjmDaLmpMonthlyAgg  # or equivalent
  da_monthly = PjmDaLmpMonthlyAgg()
  df = da_monthly.load_data_with_replacement(loading_month="2026-01-01")
  DA Monthly Agg — date range
  df = da_monthly.load_data_with_replacement(
      from_date="2025-06-01",
      to_date="2026-01-01",
  )
  Key differences from MCP:
  - time_col is "datetime_beginning_utc" (not "auction_date")
  - value_col is "congestion_price_da_daily" for daily agg, "congestion_price_da_monthly" for monthly agg
  ---
  How Backward Fill Works
  The prepare_backward_fill() function (from pbase.data.dataset.replacement) uses the nodal replacement table to impute prices
   for nodes that were renamed/replaced. It builds a replacement graph, traces node lineage backwards, and fills missing
  prices using predecessor node data within the [st, et) window.
**** end of change

What you need to do:
- review and understand the change needed from current MISO porting implementation
- this is canonical; you need to change the design docs as well as code.
- update your knowledge files
- then run some tests to see the above function calls can work
- then here is another issue: do we fix pjm's loading along with miso based on current knowledge, or do we change both?

Usage
  from pbase.analysis.mtm_calculator import merge_mtm_to_trades

  trades_with_mtm = merge_mtm_to_trades(
      trades_all=trades,
      tools=tools,        # your APTools instance
      rename=True,        # rename to mtm_1st/2nd/3rd/mcp_amortized
      slim=False,         # set True to skip mtm_now_* columns (faster)
  )

1. what is scenario_name and result_date and result_hour? result date, is it the target month or the auction month?
2. so to get a single cid for one quarter, we need to aggregate many columns together right? either sum/average/normalization?
3. what does P1 and P2 mean in "P1 and P2 are missing 7 months in PY 2022", are they just features? if so we can choose to normalize or ignore them no problem
4. yes this is only for

One more thing: this is the signal we've been using TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1/R2/R3 for annual. 