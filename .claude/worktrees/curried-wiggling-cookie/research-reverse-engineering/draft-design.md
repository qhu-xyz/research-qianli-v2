                                                                                                                           
  Section 1: Pipeline Architecture Overview                                                                                          
                                                                                                                                     
  The pipeline has six stages, executed sequentially for each annual auction:                                                        
                                                                                                                                     
  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐                                                                          
  │  1. Data     │───▶│ 2. Path      │───▶│ 3. Valuation   │                                                                         
  │  Ingestion   │    │ Screening    │    │ Engine (DC-OPF) │                                                                        
  └─────────────┘    └──────────────┘    └────────────────┘                                                                          
                                                │                                                                                    
                                                ▼                                                                                    
  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐                                                                          
  │ 6. Portfolio │◀───│ 5. Bid       │◀───│ 4. Risk Model  │                                                                         
  │ Analytics    │    │ Optimizer    │    │ (VaR/CVaR)     │
  └─────────────┘    └──────────────┘    └────────────────┘

  Stage 1 — Data Ingestion: Pull MISO DAM LMPs (energy, congestion, loss components by node), network model data (PTDFs, LODFs,
  constraint limits), generation fleet data (capacity, fuel type, heat rates), planned outages, and load forecasts. This is the
  foundation everything else builds on.

  Stage 2 — Path Screening: Score every candidate source-sink pair using a multi-factor signal model. Factors include: historical
  congestion rent (mean, median, percentiles), constraint binding frequency and severity, upcoming topology changes (new lines,
  retirements), generation additions/retirements near key nodes, and seasonal load patterns. Rank paths and select the top N for full
   fundamental valuation.

  Stage 3 — Valuation Engine: Run a DC-OPF model to estimate forward congestion value for each screened path across the annual
  delivery period. This is the core intellectual property — it translates network fundamentals into dollar values per MW.

  Stages 4-6 handle risk quantification, bid generation, and post-auction analytics. I'll detail each in subsequent sections.

  Does this high-level architecture look right before I go deeper into each stage?
