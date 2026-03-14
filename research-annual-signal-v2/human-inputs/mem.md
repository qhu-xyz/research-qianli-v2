1. avoid null fill unless extremely necessary, no default value usage, no try catch. everything needs to be explicit. When moving into production code, raiseError whenever possible to avoid gaps.
2. write as many test cases and assertions as possible


be precise when answering:
1. per (year, round, ptype), are we including around 4-5k constraints? 
1.1 so one constraint per quarter right? one direction also?
1.5. will we get a blend baseline between predictive features and historical features? how similar will our baseline be w.r.t 6.2b's baseline
1.6. for each of the constraints, will we get a real DA value for that quarter? (i mean ground truth), what pct will have missing values? is the reason we have missing values it has sp 0 or just missing?
2. will we recompute the features from scratch? what features do you propose to include? examine them one by one
- explain how they are built
- analyze how possible that we need to fillna or the feature will not be consistently available

explain how you are doing training, testing, and holdout.

explain how many metrics you will include and examine them one by one
- are we building metrics to compare model in the same universe?
- are we building metrics to compare model in the different universes?
- when juding do not just use vc@20. be comprehensive
- are we also following the gating.promotinon, and registry system from previous repos?


1. can we rebuild density_mix_rank_value and density_ori_rank_value? can we blend similar to 6.2b? do u think it is valuable?
2. but for every constraint in our universe, we can have a real DA target value right? if not, gimme an example why this is not true
3. For Feature 1: Density bins: i wonder if choosing Max or other statistics is valuable. Maybe we should not be that restrictive on which features to select
4. density_signal_score: is this sth raw score or derived by density bins?
5. 2025-06's aq4 is incomplete right?
6. we need gate to detect a model's capability in detecting new binding constraints:
- how to define NB- no binding for n: if a constraint has not bound (either in annual or month) for n months, but in reality, in realized DA / target, it has bound, then this is a new binding constraint. you can take n = 6, 12, 24 (fill is needed here as perhaps that constraint does not have such long history)
Is my undersanding here correct? is there mapping issues here? ==> or if your cohort is already doing this, ignore this part

To compare different universes:
- for top 50 universe 1 and top50 universe 2 has chosen, which one is more accurate and contains more binding constraints? this idea is evident right?

1. i need a strong formula baseline for comparison, not overtuned. this avoids overfitting
2. density_signal_score: we know 100pct how this is computed or no? i worry if not, we cannot reproduce in production
3. Use 3 holdout groups (aq1-aq3) and monitor aq4
4. onpeak and offpeak are not divided in annual right? if not divided, why not blend them togehter for BOTH features and metric checks? for example, when checking NB or binding frequency, should we check both ctypes together for a window? ==> this is important
5. I don't understand your NB12_Count for example. why not just check the capability for the model to find and rank NBs?
6.  absolute VC@50 this should be a hard gate

1. it should be tuned a bit, but not overtuned
2. if density_signal_score cannot be reproduced dont use it

One discussion between me and AI:
***
1. density_mix_rank_value / density_ori_rank_value

  You asked two separate things here.

  Can we rebuild them exactly?

  - Not from the current raw annual input set I validated.
  - The raw tables we are actually using are:
      - MISO_SPICE_DENSITY_DISTRIBUTION
      - MISO_SPICE_DENSITY_SIGNAL_SCORE
      - MISO_SPICE_CONSTRAINT_LIMIT
      - bridge table
      - realized DA
  - Those do not expose ori_mean, mix_mean, density_ori_rank_value, or
    density_mix_rank_value.

  What V6.2B has:

  - ori_mean
  - mix_mean
  - density_ori_rank_value
  - density_mix_rank_value
  - da_rank_value

  Those are part of the prebuilt monthly signal, not something I have reproduced from the
  raw annual SPICE tables.

  So the precise answer is:

  - exact rebuild: no, not from the validated raw annual tables alone
  - analogous rebuild: yes

  What I would do instead

  - Build an analogous predictive blend baseline, not claim it is the same feature:
      - hist_rank = rank(da_rank_value)
      - pred_rank_1 = rank(density_signal_score)
      - pred_rank_2 = rank(severe_density_bin_aggregate) such as bin_100_mean or a small
        density composite
  - Then define a formula baseline similar in spirit to 6.2B:
      - example: 0.60 * hist_rank + 0.25 * score_rank + 0.15 * tail_bin_rank

  Is that valuable?

  - Yes, as a baseline and sanity check.
  - No, I would not make the whole pipeline depend on recreating the exact 6.2B predictive
    terms.

  2. “For every constraint in our universe, can we have a real DA target value?”

  Numerically, yes.

  - After the left join, every row gets a numeric target because null is filled with 0.0.

  Semantically, not always a true fully observed target.

  - A 0.0 label means: “no mapped positive DA label for this row’s branch_name in the target
    quarter.”
  - That can mean:
      - truly did not bind
      - or bound in DA, but the relevant DA constraint_id had no bridge mapping for that
        quarter

  Concrete example of why this is not fully true:

  - In 2025-06/aq2, 27.5% of positive DA shadow-price value was on constraint_ids with no
    bridge mapping.
  - Those events cannot be attached to our universe rows.
  - So some universe rows get 0.0 not because reality was definitely zero, but because the
    DA event was invisible to the mapping.

  One more precision point:

  - our current row unit is constraint_id
  - labels are attached by branch_name
  - multiple constraint_id rows can share the same branch_name
  - so multiple rows can inherit the same branch-level target

  That is why I keep saying the current design is “one row per constraint_id per quarter,”
  not “one unique economic branch per quarter.”

  3. Density bins: should we use max or other stats?

  Yes. I would not restrict this to quarter-mean only.

  What I recommend:

  - for each selected bin, build multiple quarter-level summaries:
      - mean
      - max
      - std
      - hit_rate_eps: fraction of outage-date rows where bin > epsilon
      - maybe top2_mean or p90 across outage dates instead of pure max

  Why:

  - mean captures persistent signal
  - max captures stress-case signal
  - std captures instability / regime sensitivity
  - hit_rate captures frequency of nontrivial signal
  - top2_mean is often safer than pure max if max is noisy

  Would I be restrictive on which bins?

  - not at the start
  - I would start with a moderate grid, not a tiny hand-picked set
  - example:
      - negative tail: -100, -50
      - center: 0
      - positive mid: 60, 70, 80
      - positive tail: 90, 100, 110, 150
  - then prune by importance and correlation

  Availability / fillna

  - for tested 2025 aq1-3, the selected density bins had 0% nulls
  - so these are stable features and should not need fillna in normal operation

  4. density_signal_score: raw or derived?

  Derived.

  Precisely:

  - it is derived downstream from the density distribution
  - it is not a separate raw measurement
  - it is a scalar summary produced from the density-distribution pipeline

  So:

  - rawest predictive source: density_distribution
  - derived scalar summary: density_signal_score

  5. 2025-06/aq4 is incomplete, right?

  Yes.

  As of March 12, 2026:

  - aq4 for planning year 2025-06 covers:
      - 2026-03
      - 2026-04
      - 2026-05
  - realized DA for 2026-04 and 2026-05 is not fully available yet
  - so 2025-06/aq4 is not fully evaluable right now

  6. New-binding gate

  Your understanding is basically correct, with two caveats.

  Your proposed definition

  - if a constraint has not bound in annual or monthly realized history for the last n
    months
  - but does bind in the target realized DA period
  - then it is a new binder for horizon n

  That is a good definition.

  My refinements

  - define it on branch_name, not raw constraint_id
      - because our history and labels are effectively branch-based after mapping
  - use combined annual + monthly realized DA history
  - report multiple horizons:
      - NB_6
      - NB_12
      - NB_24

  Caveat 1: mapping

  - yes, mapping issues matter
  - if historical or target DA events are unmapped, a branch can look “new” when it is
    really “unmapped”
  - so every NB metric should be reported alongside the quarter’s Stage 1 unmapped rate

  Caveat 2: censored history

  - for older eval years, long windows like 24 months can be partially censored by limited
    history availability
  - for 2025-06 holdout this is less of a problem because enough history exists

  So I would keep the cohort framework, but strengthen it with explicit NB horizons:

  - NB_6
  - NB_12
  - NB_24

  And yes, these should be gated / monitored explicitly.

  Cross-universe comparison: “top 50 of universe 1 vs top 50 of universe 2”

  Yes, that idea is valid, but only if you compare on the right economic unit.

  - distinct binding branches found in top 50
  - total realized SP captured by top 50
  - precision of top 50
  - recall of top 50 against all mapped binders

  But one critical warning:

  - in the current raw design, many constraint_id rows map to the same branch_name
  - so a top-50 row list can contain duplicates of the same economic branch

  Therefore:

  - do not compare universes by raw top-50 rows
  - compare them by deduped branch_name shortlist
  - or collapse the modeling universe to branch level before comparison

  So the right cross-universe comparison is:

  - top-50 distinct branch_names from universe 1
  - top-50 distinct branch_names from universe 2
  - then compare:
      - SP captured
      - number of positive binders found
      - overlap with mapped target binders

  That is the clean way to judge which universe is actually better.

  If you want, next I can give you:

  1. the exact baseline stack I would implement first
  2. the exact feature table with build rule, unit, fillna rule, and risk for each feature
  3. the exact NB metric definitions and reporting schema


› if density_signal_score cannot be reproduced dont use it


• Agreed.

  If reproducibility is a hard requirement, we should not use density_signal_score anywhere:

  - not as a feature
  - not as a universe filter
  - not in a baseline blend

  Precise implication:

  - the current score >= 0.001 universe definition is out

  What replaces it:

  - use raw-density aggregates such as:
      - selected bin mean
      - selected bin max
      - selected bin std
      - nonzero / exceedance hit-rate across outage dates
  - define universe from those raw aggregates, for example:
      - keep rows where max(bin_70, bin_90, bin_100, bin_110) exceeds a threshold
      - or keep top N constraints by a raw-density composite score

  What stays:

  - constraint_limit
  - da_rank_value
  - bf_6, bf_12, bf_15
  - bfo_6, bfo_12

  What changes:

  - all size estimates based on density_signal_score >= 0.001 are no longer valid
  - the validated prototype I ran used density_signal_score, so I would need to rerun it
    with a raw-only universe rule

  My recommendation:

  1. Drop density_signal_score completely.
  2. Define a raw-only predictive baseline from density-bin aggregates.
  3. Define a raw-only universe rule.
  4. Recompute universe sizes, label rates, and holdout metrics from that rule.

  If you want, I can do the next concrete step:

  - propose 2-3 raw-only universe rules,
  - run them on 2025-06/aq1-3,
  - and tell you which one is closest to the coverage/size tradeoff you want.


› i was confused on your point about constraint vs branch name. for the dataset, my
  understanding is: whatever mapping we use, what we should have per row is a constraint
  and target is how much it binds in that quarter. is that not correct?


• Your high-level understanding is reasonable, but there is one important ambiguity in the
  word “constraint.”

  The clean statement is:

  - each row should represent one prediction unit
  - the target should be “how much that unit bound in the target quarter”

  The issue is that in the current raw data, we have two different candidate units:

  - constraint_id from SPICE
  - branch_name from the bridge / DA-mapped side

  Those are not one-to-one in the current pipeline.

  What is true right now in the validated raw prototype:

  - predictive raw features are naturally keyed by constraint_id
  - realized DA target is naturally mapped to branch_name
  - many constraint_ids map to the same branch_name

  Concrete 2025 examples:

  - aq1: 3,323 row candidates by constraint_id, but only 1,346 unique branch_names
  - aq2: 3,292 rows, 1,272 unique branch_names
  - aq3: 2,372 rows, 972 unique branch_names

  So if we keep one row per constraint_id, then many rows share the same quarter target
  after the bridge mapping.

  That means:

  - yes, each row has “a target”
  - but not necessarily a unique constraint-level target
  - often it is a branch-level target copied onto multiple constraint_id rows

  So your sentence is correct only if we first decide what “constraint” means operationally.

  There are two coherent options:

  1. Row unit = constraint_id

  - target = quarter DA binding value of the mapped branch_name
  - consequence: duplicate targets across multiple rows
  - predictive features stay natural
  - historical/label features are duplicated across rows sharing a branch

  2. Row unit = branch_name

  - target = quarter DA binding value for that branch
  - consequence: target unit is clean and unique
  - but raw predictive features from multiple constraint_ids must be aggregated to branch
    level first

  My view:

  - conceptually, your intended design is closer to option 2
  - technically, my first prototype used option 1 because raw features come that way
  - if we want the row semantics to be exactly “one constraint with one quarter target,” we
    should probably collapse to one unique branch-level row, unless you explicitly want
    multiple SPICE constraint_id variants per economic branch

  So the precise answer is:

  - your idea is correct in principle
  - the current raw pipeline does not yet satisfy it cleanly if we use constraint_id rows
  - because the target is not truly unique at the row level under the present mapping

  If you want, I can next show you the exact pros/cons of:

  - staying at constraint_id
  - collapsing to branch_name
  - or using a hybrid rule to choose one representative constraint_id per branch per quarter
***

Is this correct?
" The deeper blocker is structural. In annual raw data, one economic branch often has many
  raw SPICE IDs:

  - annual raw IDs with bridge entries: 12,876
  - unique bridged branches: 4,222
  - branches with more than one raw ID: 2,546 (60.3%)
  - median IDs per branch: 2
  - max IDs per branch: 45"

  then is my understanding correct?
  - several constraints may map to the same branches
  - if you use bridge, maybe diff constraint have same branch, thus same target value
  - if we use this is it better?
  1. for direct const - > const, if available, use it. that is more accurate than briding?
  2. if no direct DA const target, fall back to bridge. 



1. constraint -> branch mapping as target is the way to go. but is this correct? 
- we keep the (constraint, quarter, year) design
- but for target, we use branch's total DA as target.
2. also notice that when doing prediction, if many constraints are binding corresponding to one monitor line, include only constraints that have the most signifant shift factor differences -> can you find corresponding code in /home/xyz/workspace/psignal/notebook to support this?
- find logic here and verify what I'm saying makes sense 

under this design
1. so each row is a constraint but target is branch-level?
2. during prediction, we rank by branch's score right?
3. assume we have 300 top branches. then after deduping assuming we map to 600 constraints. then pick top 500 and do pct 20 per tier?

is that 1000 pct the design?


1. so each row have a constraint and a branch DA as target?
2. what we eventually produce are ranked constraints. when we say dedepulicate we do not mean dedup at the branch-level, right?

### after deciding the branch -> branch model, we have some more detials to decide

What I mean is 
- we can now assume we are not touching level-1 collapse
- however, we need to be carefuly about level-2.
1. which levels/numbers to include? are we only choosing from ["80", "85", "90", "100", "110"] like the annual loader or if you can reproduce, use other levels?
2. which features to extract? remember, we do not have a lot of data. you might need to do some probing on which features to include. 

Taking a look at feature: prob_exceed_110
Assume we have two constraints, A mnonitor line / branch name, B&C contingency. :
A:C with prob_exceed_110? .3
A:B with prob_exceed_110? .1
When doing aggregation, are we taking max? if so, we are assuming the max impacts the most. but what if in real data, lots of branchs with max pct over .3 do not bind at all, this feature might not be the best. 
if we take min - will we face similar risk?

***
do some research using some previous years (do not touch holdout data) and decide which to include


obvious questions we should answer:
1. are we dividing into off and onpeak?
2. are we doing branch to branch dataset building?
List your potential feature set and how you want to build them 
List your metrics and their definition at branch-level


## Recommended setting
› obvious questions we should answer:
  1. are we dividing into off and onpeak?
  2. are we doing branch to branch dataset building?
  List your potential feature set and how you want to build them
  List your metrics and their definition at branch-level


• 1. Peak Split

  I would not build separate onpeak and offpeak models.

  I would use:

  - one branch-level model
  - combined GT target = branch-quarter sum(abs(DA shadow_price)) across both onpeak and
    offpeak
  - separate history features for onpeak and offpeak:
      - bf_*
      - bfo_*
      - bf_combined_*

  Reason:

  - raw annual density features are class-type agnostic
  - splitting the model by peak type would force a distinction the predictive input does not
    naturally have
  - we can still monitor onpeak-only and offpeak-only metrics separately

  So the answer is:

  - Training: combined
  - Feature history: split + combined
  - Evaluation: combined primary, onpeak/offpeak secondary

  2. Dataset Unit

  Yes. I would build a branch -> branch dataset.

  One row:

  - (branch_name, planning_year, aq_quarter)

  Target:

  - target_sp = sum(abs(shadow_price))
  - summed across:
      - all mapped DA constraint_ids for that branch
      - all 3 months in the quarter
      - both onpeak and offpeak

  Feature construction:

  - raw density starts at constraint_id
  - then collapse to branch_name

  Pipeline:

  1. Load raw density rows for the quarter months.
  2. Level 1: collapse (constraint_id, outage_date/month) -> (constraint_id, quarter)
  3. Join annual bridge with convention < 10
  4. Level 2: collapse (constraint_id, quarter) -> (branch_name, quarter)
  5. Join branch-level history features
  6. Join branch-level target

  ———

  Recommended Feature Set

  I would define this in two groups: initial production candidate and optional ablations.

  Initial production candidate

  1. da_rank_value

  - Build: historical branch-level cumulative abs(DA SP) over allowed lookback window, then
    rank within the training snapshot
  - Unit: branch
  - Why: strongest proven historical feature

  2. bf_6, bf_12, bf_15

  - Build: fraction of prior months with any onpeak DA binding on the branch
  - Unit: branch

  3. bfo_6, bfo_12

  - Build: same, but offpeak
  - Unit: branch

  4. bf_combined_6, bf_combined_12

  - Build: fraction of prior months with any binding in either peak type
  - Unit: branch

  5. count_cids

  - Build: number of annual SPICE constraint_ids mapped to the branch in that quarter
    universe
  - Unit: branch

  6. count_active_cids

  - Build: number of cids under the branch that pass the raw-density universe activity rule
  - Unit: branch
  - Important: this depends on the final raw-only threshold rule

  7. limit_min, limit_mean, limit_max, limit_std

  - Build:
      - Level 1: one constraint_limit per cid-quarter
      - Level 2: aggregate across cids under the branch
  - Unit: branch

  8. Density bin features, branch-level

  - My preferred initial compact set:
      - bin_80_cid_std
      - bin_85_cid_std
      - bin_90_cid_std
      - bin_100_cid_std
      - bin_110_cid_std
  - Build:
      - Level 1: for each selected raw bin, mean across outage dates and quarter months per
        cid
      - Level 2: std across cids within the branch
  - Why: in my pre-holdout branch probes, this compact std family was the best small branch-
    level density set

  Optional ablations / expansions

  9. bin_95_cid_top2, bin_105_cid_top2

  - Build: mean of top 2 cid values within branch
  - Use: first density ablation after the compact core

  10. Counter-flow / wider-tail bins

  - Candidates:
      - bin_-100_*
      - bin_-50_*
      - bin_60_*
      - bin_120_*
      - bin_150_*
  - Build same way
  - Use only if dev results show real lift

  11. Raw derived density summaries

  - right_tail_sum
  - right_tail_mean
  - peak_bin_value
  - positive_tail_mass
  - distribution_width
  - Build from raw 77 bins after Level 1, then branch-collapse at Level 2
  - Useful if you want more interpretable raw-only density summaries

  What I would not use

  - density_signal_score, unless it is reproduced cleanly from raw bins
  - all 77 bins at once
  - min as the default second density stat
  - V4.4-derived opaque features

  ———

  Branch-Level Metric Set

  Let:

  - y_i = branch target SP for row i
  - binder = y_i > 0
  - rank by predicted score descending within each (planning_year, aq_quarter)

  Primary within-universe metrics
  These compare models on the same branch universe.

  1. VC@K

  - Definition: sum(y_i in top K) / sum(y_i over all branches in the modeled universe)
  - Meaning: fraction of in-universe binding value captured in top K

  2. Recall@K

  - Definition: # binding branches in top K / # binding branches in universe
  - Meaning: branch hit-rate

  3. NDCG

  - Definition: standard ranking NDCG using tiered branch labels 0/1/2/3
  - Meaning: full-list ranking quality

  4. Spearman

  - Definition: Spearman correlation between predicted branch rank and actual branch target
    rank
  - Meaning: global ordering quality

  5. Tier0-AP, Tier01-AP

  - Definition: average precision for top relevance tiers
  - Meaning: how well the model concentrates the strongest binders near the top

  Cross-universe metrics
  These compare models when the universes differ.

  6. Abs_SP@K

  - Definition: sum(y_i in top K) / total DA binding SP for the quarter
  - Denominator uses all DA binding SP, not just in-universe SP
  - Meaning: actual trading value capture across different universes

  7. Abs_Binders@K

  - Definition: count of top-K branches with y_i > 0
  - Meaning: how many real binders the shortlist contains

  8. Precision@K

  - Definition: Abs_Binders@K / K
  - Meaning: branch shortlist purity
  - I would report this even if it is not a promotion gate

  New-binding metrics
  These test whether density adds value beyond history.

  9. NB12_Recall@50

  - Definition: among branches that had no binding in either ctype for 12 months but do bind
    now, fraction found in top 50

  10. NB12_SP_Capture@50

  - Definition: branch SP captured from NB12 binders in top 50 / total NB12 branch SP

  11. NB12_Median_Rank

  - Definition: median predicted rank of NB12 binding branches

  Also monitor:

  - NB6_*
  - NB24_*

  Class-type monitoring metrics
  Same model, alternate labels.

  12. onpeak-only VC@50, Recall@50

  - Use branch target built from onpeak-only DA

  13. offpeak-only VC@50, Recall@50

  - Use branch target built from offpeak-only DA

  These are monitoring metrics, not primary gates.
  ———

  1. treat ptype, ctypes seriously. do not confound or just assume.
  2. ALL pieces used should be reproducible. You said you used "the DA cache is in stage5-tier" is this reproducible?

  ## phase 3 possible directions
  Issues:
  1. currently our model performs poorly in the NB-related metrics
  2. we have not explored all options for model selections
  3. we have not exhausted the target definition

  1. NB-related metrics
  - need to find out which **predictive features** are most relevant for NB constraints. We should focus more on the predictive features here: for example, if a constraints has not bound for prev 24 months, but the ground truth label shows it will bind in aq2, which features are helpful?
  - there can be many NB definitions. For example, constraints has not bound for prev 6/12/24 months perhaps will have different indicators, choose diff window size and analyze

  2. model selection
  - we tried blending. have you treid linear regression/logistic regression?
  - build a model just using predictive features and see how it can **augment** current champion. it has been shown that v0c misses NB entirely. We need a model that can:
  a. at least match v0c in general statistics/metrics
  b. do MUCH MUCH better in NB-related stats
  
  3. target definition
  - for ML/lamdrank models, are you dividing the shadow prices into bins? which method have you used? 
  - are you dividing targets into bins based on qcut or cut by sp? what difference does it make? inspect.


what does those columns/features mean? explain avec exemples
- NB12_C, NB12_SP, Dorm_C, Dorm_SP
- when you say fewer NB captures, (0.3, vs 1.0) what does 0.3, 1, mean? capture percentage? does it mean that 
- in general I dont get why you are measuring for NB. or how good the new models are. For instance: 
  - are you training the NB-model only on nNB data? meaning that if a branch has bound before, it will not be in train?
  - what does the results mean? for 0.0021 NB12_SP, does it mean we only capture 0.21's total shadow?


-   NB12_SP (NB12_SP@K): The ratio of NB12 shadow price captured in top-K vs total NB12 shadow price
  available. If all NB12 binders in the universe generated $1M total SP, and the ones that landed in
  top-K generated $5k, then NB12_SP = 0.005 (0.5%). ==> we are producing sth ratio like this, and it just means that the new model is NOT affective at all. 

- Holdout Results — K=100, R=15 in this table, why the dorm_c count is either 0 or 15?
-  Dorm_SP: The raw dollar amount of realized shadow price from dormant branches in top-K. -> here top-K means the top-K we predicted right?

- Example: In aq3 holdout, Phase 3 has Dorm_C=5, Dorm_SP=$5,352, NB12_C=2, NB12_SP=0.0147. That means:
   5 dormant branches made it into top-50. Of those 5, only 2 actually bound. Those 2 binders
  generated $5,352. The total SP from ALL NB12 binders in the universe was $5,352 / 0.0147 ≈ $364k. So
   we captured 1.47% of available NB12 value.
   ==> exmamples like this are true right? then wouldn't it suggest that the 5 branches we are picking into the universe are jsut horrible?

- What we're training: The Track B model is trained only on branches that have NO binding history
  (dormant cohort). ==> what you mean by No binding hist? within 12 month or none at all?

## improving the prediction model
- I think we ned to aim for at leas 10pct 

## metric rebuilding
- now my teammate has told me that:
a. we will include 150 tier0 constraints and 150 tier1. which means we need to look at @150 and @300. ==> from now on, use only those levels, do not use @50 100
b. in this setting, does our NB model help more? for 150, we can include let's say maybe 20, 30, 40 NB predictions. for 300, even more. lot's of areas to try.
- also we should consider this metric: 
  - the most dangerous branch are the ones with super high realize DA, let's say >50000. which model is the best at finding those? 