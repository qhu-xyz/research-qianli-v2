##
1. why include both shadow_price_da and da_rank_value aren't they the same for ML?
2. for planning year 2026, do we have every bit piece of data to produce signals with rank?
3. any gap we need to fill to produce the exact type of signal like 6.1b?

## augmenting feature set
- how likely is it that we can use monthly signals?
for example, for a constraint that we are bidding in aq1, (6,7,8) is it possible to check monthly data to see if this constraint has bound in previous months? check this repo below to see how then we can build features like binding_freq_12 (12 month prior)
- but be very careful: we submit early april, so you can ONLY use monthly binding information realized <= march. Notice the difference between concepts such as auction month, market month, and period type.
/home/xyz/workspace/research-qianli-v2/research-stage5-tier
** there might be severe mapping issues that you need to figure out. 

for miso: 
1. you need to remember that we submit in quarters. are you sure you are doing: for aq2, which happens in 9, 10, 11, you are correctly using constraint's binding only <=month3? and this is helpful? then moving onto aq3, aq4, they are even further apart, are they still helpful?
2. for each constraint in annual, are you saying we CAN find its monthly track? what percent of constraints can we find and not find?

### inspect this logic to see if we can add the half monthly data into the mix
the fix aims to solve this problem:
- if we submit annual round1 at 04-10, then it cannot uses the 04-01 to 04-09 binding data of that constraint, hence why for v8 only includes data from month march and previous (check carefully for data leakage.) 
- that was for annual. but can we do similar thing for monthly signal?
""
● Found it. The bidding window loader is in pbase.data.dataset.ftr.market.base:

  - Class: BaseMarketInfo(RtoBase, ABC) — base class for loading FTR Market Info data
  - Method: get_bidding_window(auction_month, market_round, auction_type) — returns a tuple[pd.Timestamp, pd.Timestamp]
  (start, end) for the bidding window of a given auction configuration
  - Factory: get_market_info(rto) — returns the appropriate BaseMarketInfo subclass for an RTO (e.g., PJM, MISO)

  Usage would look like:

  from pbase.data.dataset.ftr.market.base import get_market_info

  market_info = get_market_info("PJM")
  start, end = market_info.get_bidding_window(
      auction_month=pd.Timestamp("2026-04-01"),
      market_round=1,
      auction_type="monthly",
  )
""
u know what i mean?

## moving on to pjm, restructuing code base
- now all our code base focuses on 

1. inspect data availability