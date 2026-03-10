### concerning data leakage

Now think about this:
"
 now think about this. at 2025-03, do we actually have 2025-02's binding data? for market month march, we submit around mid of
  feb. is this leaky data or no?


• I’m checking whether the repo assumes f0 is scored after month-end or before the target month starts. If the business timing is
  mid-February for March delivery, then using 2025-02 realized outcomes would indeed cross the temporal boundary.

• Explored
  └ Search V6.2B signal runs|~5th of month|mid of|auction month|market month|f0|submission|submit|delivery|realized DA is
available in
           scripts
    Search load_train_val_test\(|target_month|eval_month|auction_month|market_month in ml

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Yes. Under the timing you described, this is leaky.

  For a March 2025 f0 decision made in mid-February 2025:

  - You do not know full-month 2025-02 realized DA binding outcomes yet.
  - But the current binding_freq_* code for target month 2025-03 uses all prior calendar months < "2025-03", which includes 2025-
    02.

  That logic is here:

  - scripts/run_v9_binding_freq.py:87
  - scripts/run_v10_pruned_features.py:66
  - scripts/run_v10_variants.py:61
  - scripts/run_v10e_holdout.py:55

  So for 2025-03, the code effectively does:

  - binding_freq_1: uses 2025-02
  - binding_freq_3: uses 2024-12, 2025-01, 2025-02
  - binding_freq_6: uses 2024-09 .. 2025-02

  If the real submission date is around mid-February 2025, the latest fully known month should be 2025-01, not 2025-02.

  So the production-safe version for March 2025 should be more like:

  - binding_freq_1: use 2025-01
  - binding_freq_3: use 2024-11, 2024-12, 2025-01
  - binding_freq_6: use 2024-08 .. 2025-01

  The repo explicitly assumes the opposite timing. It says:

  - scripts/run_v9_binding_freq.py:380
  - registry/v9/NOTES.md

  Those files assume the signal is generated after the previous month closes. If that assumption is wrong for f0, then v9 and the
  v10* family are using information they would not have at decision time.

  So the answer is:

  - Ex post monthly backtest interpretation: not leakage
  - Actual f0 auction timing you described: yes, lookahead leakage

  This is likely the main reason the binding_freq_* lift looks so strong.
  "

  - if this is true, then we should not build binding this way, or we should use this constraints binding freq at uses all prior calendar months < "2025-02-12", before we submit. => this is a logic we do not yet have.
  - **also if this is true, the way we've constructed the whole experiment is leaky. because for target month 2025-03, we are using the month 2025-02's data, but actually we only know half of the data at the submission time.**
*****
you get what i mean?

## f1 and onwards
### f1

now to avoid leak, the data should be managed in a diffrent manner.
#### on dataset
for target month 2025-03, auction happens at mid 2025-01, which means we can only use data <= 2024-12 right?

#### on the features
if that is the case, check every features that we've used in v10e-lag1 to see which ones are potentially leaky for f1.

## example:
for f1, 2025-03,
training data goes until 2024-12 right? ==> this is for which rows to include
the frequency features for **rows of 2025-03, market month** should also only use bidning frequency data <=2024-12 right? ==> this is for how to calculate the features
the examine other features, any other thing we should be aware of to avoid leakage?

for f1, 2025-03.
while building feature
for feature frequency-last-n:
the last month should be 2024-12 right?
and the last 

i mean but this is correct right?

i see what you mean. then:
- if each row is (auction month, ptype), with market month implied. then:
1. if market = 2025-03, ptype = f1, auction month = 2025-02. auction happens mid jan.
2. then the last available data would be auction month = 2024-11, market = 2025-12?
for frequency features, include the latest month to see if it binds by using auction_month - 2?


## one more thing on f1 and onwards

for some months f1 are missing. if we only use auction month's previous **usable 8 months** we might get little data, as i think may, june do not have f1? so for some months, we might get less than 8 months if we use a (-10, -3) look back or something like that. can we reiterate this as using 8 avaiilable months without leakage? are we already doing this in f0?

several important things:
- i'm not sure if (.85, 0, .15) is the best mix for updated v0 (v0 uses 0.6, .3, .1 right? what about f1? we might need to udpate this.
- after fix, for (auction month =2025-02, ptype f1)
- we are only including rows (auction month = 2024-11, ptype f1) right or wrong?
for features binding frequency for each row:
for (auction month =2025-02, ptype f1), we can only use that constraint's binding data <=2024-12?

those are completely different concepts in my view. one about which data to include during training, the other about how features for each row are built.

if the above are true, is the implementation plan 100pct the same?

## are my questions relevant, and if they are, are they addressed?
1. in holdout if month 2025-08 f1 is in hold out, then does it know that some seemingly non-leaky months before are not there and still fetches 8 months of data to train?
2. are we using caching to speed up training? 
3. are we using 8 month train, using the same metrics and gates? will we recalibrate flooring based on v0?


### on deployment
1. based on our f0 pipeline, assuming we only update f0, are there are real gaps involved?
2. i don't know exactly what is cached and how is it cached but do you think let's say in the time now (2026-03) i want to run our own pipeline to produce tier0 (rank then divide into tiers), what data is missing? 
3. in other words, i think since 6.2b is imply a formula,  itcan run for whatever month, but is there a gap for our ML pipeline, say v10-lag1?


## potential data gap:

- for f0: "no realized DA for Jan 2026 yet"


now f1 is fixed. now let's expand your f0-deployment to cover f1 also. what i want essentially: I do not know for sure how 6.2b exposes the signals, but what i want
  for our version if we call it 7.0 is: use the udpated ML method and produce EXACTLY the same result. does my intention make sense, is it on par with your
  understanding?


now for v7.0 deployment

** MY goal:
- what i want: use the udpated ML method and produce EXACTLY the same result for f0& f1, reuse 6.2b for other ptypes.
- concerns: you did not cache enough data for current prediction. for instance you said: for f0: "no realized DA for Jan 2026 yet."
    - check: for 6.2b, what is the last month for f0, and f1? if we can inference for 6.2b, then we should be able to infer for the same month for 7.0 right? Not talking about validation, just inference. --> if not, it means something aboout our feature building is not correct.
- speed. I'm not sure how valid the recommendation is in the reference: /home/xyz/workspace/research-qianli-v2/research-stage5-tier/codex-review/f0-f1-deployment.md. i fear that since we need inference from ML model, speed is slower than 6.2b? but can it be relatively fast?



Here's the picture:          
                                                                                                                                            
  V6.2B signal data (what 6.2B uses to produce its formula score):                                                                          
  - f0: through 2026-03 (latest)               
  - f1: through 2026-03                                                                                                                     
                                                                    
  Spice6 density (prob_exceed_110, constraint_limit):
  - Through 2026-03 — matches V6.2B exactly

  Realized DA cache (what we use for binding_freq features):
  - Through 2025-12 (onpeak + offpeak)
  - Missing: 2026-01, 2026-02, 2026-03

  So V6.2B can produce a signal for auction month 2026-03 (f0 or f1) because it only uses its own parquet features (da_rank_value, density
  scores). Our ML model v10e-lag1 cannot produce a prediction for 2026-03 because:

  For f0, auction month 2026-03 (market month 2026-03):
  - With lag=1, binding_freq needs realized DA through 2026-01 (months < 2026-02)
  - We only have through 2025-12 — missing 2026-01

  For f1, auction month 2026-03 (market month 2026-04):
  - With lag=2, binding_freq needs realized DA through 2025-12 (months < 2026-01)
  - We do have through 2025-12 — this actually works!


what I mean is: when we deploy a version like; TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1, or V7.0 in future, we are using the **cached parquets right?** so to actually use the signals, there is no ML at all and thus speed is not an issue right?


move to /opt/temp/qianli/spice_data/miso