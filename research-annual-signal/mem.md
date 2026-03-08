# task: generate annual signal based on what we already have

1. check /opt/data/xyz-dataset/spice_data/miso to see where signal for miso annual lives.
2. I have not read that code base, it is just my teammate telling me that this is where I can fetch miso data for signal prediction. My understanding of the signal generation pipeline:
- this is about generating annual constraints' tiers, to predict which constraints will bind in annual rounds (aq1 - aq4)
- in the folder above, we will find annual signals/features for constraints. features might include
    - flow information, prob_exceed_n etcetc for each constraint

## my vision for how the pipeline can be setup:
1. find all annual rounds' constraints information from 2020-2026
2. for those data, find necessary feature columns
3. find ground truth column (for that auction, for that conostraint, what is the true shadow price?) this should be separated from the historical DA shadow price columns
4. then build our model whether using rule-based idea or ML to **predict given any constraint in future, will it bind** I'd like the same tier number as 6.2B. tier0 is the most heavily binding, tier4 is the least. Number of constraints should also be not that high and mimic true number of binding constraints.
6. then set up the repo similar to stage4 repo, with promotion gates and model registry. Benchmark can be a simple blend then later iterations are ML iterations.

## reference
- /home/xyz/workspace/research-qianli-v2/research-stage4-tier/README-from-stage3.md
- /home/xyz/workspace/research-qianli-v2/research-stage4-tier/README-from-stage3.md

those above are for non-annual auctions but you can learn from them.


** NOTICE previously there are a LOT of false information, regarding which features are real or not.
/home/xyz/workspace/research-qianli-v2/research-stage5-tier/stage5-handoff.md -> my teammate has just verified that shadow_price_da is NOT leaked column. do thorough checks to see which features are actual features, **

Now do you understanding 100pct what you need to do?
Let's first get a solid grasp on what this task is about, and what data we have.


0. **can we produce a proper ML setup?** meaning NO leakage features, future looking targets.
1. check what features does 6.1b uses
2. check some other repo to see for our new version, if we have new features available
3. check constraint space - does 1 and 2 above have the same rows? meaning that are we getting the same or roughly the same constraints per round, same as 6.1b, but there is a repo somewhere that provides new features? (if there are, find me that repo)


one more question: 
- can you reproduce the 6.1b pipeline to a tee to set up a benchmark?
- has our team wrapped 6.1b into a wrapper as a signal called MISO. ... 6.1b sthsth? can you locate the exact pipeline and can we reproduce the pipeline result as our v0, benchmark?

have you already
1. locate the signal wrapper of 6.1b that calls it (or the latest version of annual signal)
2. regenerate the exact pipeline that generates 6.1b
so that they match to a tee? intention: our pipeline should be using the exact same old stable structure and build on top of it.

do you understand what I mean?

here is what i mean: 
1. TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1 is the wrapper
2. if we use our pipeline and use the CORRECT target, 
can 1 and 2 produce the same result?

do you understand what I mean, does it make sense?

what i mean is:
1. can our pipeline produce the EXACTLY the same tier prediction as the wrapper?
2. if so, the correct target would be the actual binding shadow price for that target month. can we find it?
if 1 and 2 is true, i think we are fine, right?


1. make sure you are setting up model registry, gates, promotion planning
2. v0 should be the baseline -> use ground truth to evaluate and set up flooring in promotion etcetc
3. then iterate on top of v0. since we do not have that many groups, ML may not be a must. but setting up the correct and production-exact baseline is a must in v0.

you mentioned:

"
The density distribution parquet has the exceedance probability columns (80, 85, 90, 100, 110 etc.)   
  and the auction_type column. The density signal score has just a single score column. The plan's
  data_loader.py was looking at the wrong file. Let me also check if there's a flow_direction column    
  here. 
● The spice6 density scores didn't have exceedance columns. Let me check what columns the density score
  parquet actually has.    
● The density signal score parquet has a score column, not separate prob_exceed_* columns. And there's  
  no auction_type column in its schema — there's auction_month, market_month, market_round, outage_date.
   Let me check the density distribution which may have the exceedance probabilities.
"