# Fix pjm's first monthly auction's mcp prediction

## issue
- pjm's annual happens in apr/may and the trades will be cleared late in may. there are four rounds. for instance, the lastest round will be cleared early in may. each path contains the volume for a whole year
- pjm's first monthly auction will be hosted 
- then comes monthly auction, auction happens at around 05-15-2025 for june auction. in this auction, we will bid f0 (market month june, 2025) till f11 (market month may, 2026)
- predicting each market month's mcp is important.historically speaking, we use annual_mcp/12 for each month
- however this is highly inaccurate & we suspect that the mcp distribution will be heavily leaning towards earlier months after the june aucntion clear (notice: we are ONLY targeting june auctions from f0 till f11)
- in history you might see sth like: path A has annual mcp 120. june f0 is 50, july 40, ...., next year's may -10, summing up to sth like 130 (sum won't match 120 as they are different auctions and public opinion shifts)
- goal: retrieve real data, do more research, and let's formulate a way to define mtm's **percentage distribution** specifically for this june round in pjm.

1. find pjm's data first, find last round's annual clearing. use /pbase skill if you need
2. use year 2020-2022
3. then check the same path's f0 - f1 mcp for the june auction only
4. do research

- report back to me how you understand the problem and what data you are using.
- then proceed. 

## trades 
- path-level change to previous year's breakdown for pjm's june auction

## check sth else
- this imprvoement for mtm_1st_mean, is it statistically signifant?
- you used a global scaling of 88pct right? what if you don't scale?
- node-level adjust:
    - did you remember we talked about node-level tracing back to previous year's june auction and extract f0-f11's distribution?
    - does this series of ideas (it also involves scaling so there are at least 2 implementations to this idea) apply for high risk paths?
    - coverage: you reported coverage for this node-level tracking back is really low in dataset. this is concerning.