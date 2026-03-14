# Goal
Produce a 7.0b constraint tier system similar to miso's 7.0

## background
Check the repo below to see context:
- /home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred
- /home/xyz/workspace/research-qianli-v2/research-stage5-tier
- /home/xyz/workspace/research-qianli-v2/research-miso-signal7
similar to miso, pjm's 7.0 changes only f0, f1, and carry over f2p (for pjm, f2p is f2 - f11.)

pipeline:
1. you have to find the original source of pjm's 6.2b. find the features and **true target column.** be careful: you have to distinguish which is which. Run tests and read everything necessary if needed.
2. then find the original blend for 6.2b; is it the same as miso, (.6, .3, .1)?
3. miso's 7.0 has included new spice6 signals?
then we run an experiment using the exact structure as stage5:
4. to run pjm's experiment similar to stage5, we need to set beforehand: find datasource, determine which months to include in the training data, build features, set v0, gates, promotion rules, metrics,then find better blend, add ML features.

things to notice
1. data leak, this is the most important. There can be no data leak. You can see previous discussions to see how to avoid leak: the dataset should be excluded/included based on ptype (we are doing f0, and f1 differently, organize the folder well); the features should be built with cutoff time for real da shadow.
2. pjm has onpeak/dailyoffpeak/wndonpeak different from miso, so you need to organize your registry a bit differently. Each (ptype, ctype) combo should be treat different and get their own models.


