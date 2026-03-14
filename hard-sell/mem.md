# generate hard sell bid prices for trades

- please refer to /home/xyz/workspace/pmodel/notebook/ql/2026-mar/miso to understand how to attach bid prices to trades:

- trades location:
"please append bid prices for these trades to sell, just like previous auctions:"
['/opt/temp/shiyi/trash/sell_hard_miso/miso/f0_prod1_auc2604/trades_onpeak_f0_auc2604.parquet',
 '/opt/temp/shiyi/trash/sell_hard_miso/miso/f0_prod1_auc2604/trades_offpeak_f0_auc2604.parquet',
 '/opt/temp/shiyi/trash/sell_hard_miso/miso/f1_prod1_auc2604/trades_onpeak_f1_auc2604.parquet',
 '/opt/temp/shiyi/trash/sell_hard_miso/miso/f1_prod1_auc2604/trades_offpeak_f1_auc2604.parquet']

1. there is no specialized sell, all you need to do is to produce the normal sells. 
2. this is for april auction. 
- you need to migrate the scripts into the repo you are in now.
reuse the params for miso march.
- follow whatever conventions you see there and migrate.

Now before you implement that, let's make sure you understand you know the details. how many which files to generate?

## picked trades:
443664:ALDRICH-FIFTHST FLO COON CREEK TR9:NSP34X04|-1|auc ==> change sp to 3000

follow the same procedure as the old notebooks to:
- pick trades sf > 0.07 or sf * vol > sthsth
- use this sp to change mtm, then apply the same pipeline 
save to: /opt/temp/qianli/miso/apr_{version}/trades_to_sell_miso_auc2604{version}_exposure_controlled_{period_type}_{class_type}_1.parquet
Is my instruction clear?
gimme what you will do step by step
this applies only for f0 onpeak/offpeak. check beforehand how many trades you will pick.