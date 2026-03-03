## purpose of this repo
- fork from https://github.com/xyzpower/research-spice-shadow-price-pred. There are two pipelines. /home/xyz/workspace/research-qianli-v2/research-stage1-shadow is focusing on the 1st pipeline, classification. this repo focuses on the next, shadow price prediction
- structural organization:
    - I want to port over the 3-iter-per-report engineering structure from the 1st pipeline
    - scan the repo andthe import pieces
    - some notable mentions: memory system dsign, functionalities of each agents with roles and accesses to files, and registries of promotions and gates

## dataset & metrics
- metrics is what makes or breaks this repo. 
- notice the business incentive file: /home/xyz/workspace/research-qianli-v2/research-stage1-shadow/human-input/business_context.md. align our metric choices and dataset to that goal.
- previous repo chooses stage 2 to only include data from stage 1. how can we replicate this? does it mean that we need to port over stage 1's logic before using stage 2? what would you suggest?

- now before planning, dive deep into the repos and let's first brainstorm.