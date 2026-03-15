# Segmented Band Coverage Analysis

## Executive Summary

The aggregate P95 coverage numbers (94-95%) hide a **massive direction asymmetry**:

| Segment | R1 P95 err | R2 P95 err | R3 P95 err |
|---------|-----------|-----------|-----------|
| **prevail** | -0.1pp | +0.9pp | +0.8pp |
| **counter** | -2.4pp | -14.2pp | -16.3pp |

**Counter-flow paths are severely under-covered in R2/R3** (P95 coverage ~79-81% vs 95% target).
R1 counter under-coverage is milder (~-2.4pp avg) because nodal_f0 residuals are already wide.
R2/R3 counter under-coverage is catastrophic (~-14 to -16pp) because MTM-based baselines
have small residuals when prevailing but large residuals when the market reverses direction.

---

## 1. Direction Breakdown

**Definition:**
- **Prevail**: sign(baseline) == sign(mcp) — market cleared in the predicted direction
- **Counter**: sign(baseline) != sign(mcp) — market reversed vs prediction
- **Zero baseline/mcp**: baseline or clearing at 0 (negligible count)

### R1 (baseline=nodal_f0, 4 bins)

| Quarter | Direction | Count | % | P95 cov | P95 err | P50 cov | P50 err | Mean |res| | P95 |res| | P95 width |
|---------|-----------|------:|--:|--------:|--------:|--------:|--------:|----------:|----------:|----------:|
| aq1 | prevail | 118,205 | 80% | 94.9% | -0.1pp | 52.9% | +2.9pp | 844 | 3,403 | 5,791 |
| aq1 | counter | 27,962 | 19% | 93.4% | -1.6pp | 37.5% | -12.5pp | 635 | 2,288 | 3,498 |
| aq2 | prevail | 120,780 | 81% | 94.8% | -0.2pp | 52.0% | +2.0pp | 1,033 | 4,348 | 6,864 |
| aq2 | counter | 26,348 | 18% | 93.5% | -1.5pp | 37.8% | -12.2pp | 597 | 2,183 | 3,565 |
| aq3 | prevail | 113,671 | 83% | 95.1% | +0.1pp | 52.4% | +2.4pp | 849 | 3,436 | 5,390 |
| aq3 | counter | 22,054 | 16% | 92.4% | -2.6pp | 34.5% | -15.5pp | 564 | 2,157 | 2,923 |
| aq4 | prevail | 113,639 | 84% | 94.9% | -0.1pp | 51.4% | +1.4pp | 753 | 3,154 | 4,685 |
| aq4 | counter | 20,440 | 15% | 91.1% | -3.9pp | 33.0% | -17.1pp | 468 | 1,749 | 2,305 |

### R2 (baseline=mtm_1st_mean, 6 bins)

| Quarter | Direction | Count | % | P95 cov | P95 err | P50 cov | P50 err | Mean |res| | P95 |res| | P95 width |
|---------|-----------|------:|--:|--------:|--------:|--------:|--------:|----------:|----------:|----------:|
| aq1 | prevail | 966,759 | 91% | 95.9% | +0.9pp | 53.3% | +3.3pp | 70 | 257 | 458 |
| aq1 | counter | 91,746 | 9% | 81.1% | -13.9pp | 14.8% | -35.2pp | 73 | 253 | 196 |
| aq2 | prevail | 995,076 | 91% | 95.9% | +0.9pp | 53.3% | +3.3pp | 76 | 286 | 482 |
| aq2 | counter | 89,169 | 8% | 81.0% | -14.0pp | 13.8% | -36.2pp | 75 | 253 | 197 |
| aq3 | prevail | 916,110 | 91% | 95.9% | +0.9pp | 53.3% | +3.3pp | 67 | 246 | 424 |
| aq3 | counter | 82,638 | 8% | 79.7% | -15.3pp | 12.7% | -37.3pp | 72 | 248 | 182 |
| aq4 | prevail | 920,952 | 90% | 96.0% | +1.0pp | 53.9% | +3.9pp | 68 | 252 | 446 |
| aq4 | counter | 94,164 | 9% | 81.3% | -13.7pp | 12.6% | -37.4pp | 72 | 255 | 195 |

### R3 (baseline=mtm_1st_mean, 6 bins)

| Quarter | Direction | Count | % | P95 cov | P95 err | P50 cov | P50 err | Mean |res| | P95 |res| | P95 width |
|---------|-----------|------:|--:|--------:|--------:|--------:|--------:|----------:|----------:|----------:|
| aq1 | prevail | 1,163,472 | 93% | 95.7% | +0.7pp | 52.8% | +2.8pp | 58 | 214 | 378 |
| aq1 | counter | 85,425 | 7% | 78.9% | -16.1pp | 13.3% | -36.7pp | 65 | 222 | 166 |
| aq2 | prevail | 1,185,345 | 93% | 95.9% | +0.9pp | 52.6% | +2.6pp | 60 | 222 | 378 |
| aq2 | counter | 86,961 | 7% | 78.8% | -16.2pp | 11.9% | -38.1pp | 62 | 214 | 155 |
| aq3 | prevail | 1,109,640 | 93% | 95.7% | +0.7pp | 52.6% | +2.6pp | 53 | 189 | 337 |
| aq3 | counter | 76,773 | 6% | 77.6% | -17.4pp | 10.8% | -39.2pp | 66 | 207 | 146 |
| aq4 | prevail | 1,110,693 | 93% | 95.8% | +0.8pp | 52.8% | +2.8pp | 52 | 185 | 338 |
| aq4 | counter | 83,217 | 7% | 79.3% | -15.7pp | 11.9% | -38.1pp | 58 | 185 | 148 |

---

## 2. Direction x Class (onpeak/offpeak)

### R1

| Quarter | Segment | Count | P95 cov | P95 err | P95 width |
|---------|---------|------:|--------:|--------:|----------:|
| aq1 | prevail_onpeak | 61,202 | 95.0% | +0.0pp | 6,033 |
| aq1 | prevail_offpeak | 57,003 | 94.8% | -0.2pp | 5,531 |
| aq1 | counter_onpeak | 13,745 | 93.0% | -2.0pp | 3,815 |
| aq1 | counter_offpeak | 14,217 | 93.7% | -1.3pp | 3,192 |
| aq2 | prevail_onpeak | 62,303 | 94.8% | -0.2pp | 6,970 |
| aq2 | prevail_offpeak | 58,477 | 94.9% | -0.1pp | 6,751 |
| aq2 | counter_onpeak | 13,161 | 93.5% | -1.5pp | 3,636 |
| aq2 | counter_offpeak | 13,187 | 93.5% | -1.5pp | 3,494 |
| aq3 | prevail_onpeak | 58,029 | 95.2% | +0.1pp | 5,609 |
| aq3 | prevail_offpeak | 55,642 | 95.1% | +0.1pp | 5,161 |
| aq3 | counter_onpeak | 11,022 | 92.2% | -2.8pp | 3,100 |
| aq3 | counter_offpeak | 11,032 | 92.5% | -2.5pp | 2,746 |
| aq4 | prevail_onpeak | 57,894 | 94.9% | -0.1pp | 4,604 |
| aq4 | prevail_offpeak | 55,745 | 94.9% | -0.1pp | 4,769 |
| aq4 | counter_onpeak | 10,223 | 91.2% | -3.8pp | 2,381 |
| aq4 | counter_offpeak | 10,217 | 90.9% | -4.1pp | 2,229 |

### R2

| Quarter | Segment | Count | P95 cov | P95 err | P95 width |
|---------|---------|------:|--------:|--------:|----------:|
| aq1 | prevail_onpeak | 503,994 | 95.8% | +0.8pp | 491 |
| aq1 | prevail_offpeak | 462,765 | 95.9% | +0.9pp | 422 |
| aq1 | counter_onpeak | 46,005 | 81.3% | -13.7pp | 208 |
| aq1 | counter_offpeak | 45,741 | 80.8% | -14.2pp | 183 |
| aq2 | prevail_onpeak | 516,573 | 95.9% | +0.9pp | 483 |
| aq2 | prevail_offpeak | 478,503 | 95.9% | +0.9pp | 482 |
| aq2 | counter_onpeak | 44,100 | 80.4% | -14.6pp | 206 |
| aq2 | counter_offpeak | 45,069 | 81.5% | -13.5pp | 188 |
| aq3 | prevail_onpeak | 475,863 | 95.8% | +0.8pp | 442 |
| aq3 | prevail_offpeak | 440,247 | 96.0% | +1.0pp | 405 |
| aq3 | counter_onpeak | 40,194 | 79.6% | -15.4pp | 194 |
| aq3 | counter_offpeak | 42,444 | 79.8% | -15.2pp | 172 |
| aq4 | prevail_onpeak | 473,175 | 96.0% | +1.0pp | 447 |
| aq4 | prevail_offpeak | 447,777 | 96.0% | +1.0pp | 444 |
| aq4 | counter_onpeak | 46,638 | 81.2% | -13.8pp | 205 |
| aq4 | counter_offpeak | 47,526 | 81.4% | -13.6pp | 186 |

### R3

| Quarter | Segment | Count | P95 cov | P95 err | P95 width |
|---------|---------|------:|--------:|--------:|----------:|
| aq1 | prevail_onpeak | 604,323 | 95.8% | +0.8pp | 409 |
| aq1 | prevail_offpeak | 559,149 | 95.6% | +0.6pp | 345 |
| aq1 | counter_onpeak | 42,765 | 78.4% | -16.6pp | 183 |
| aq1 | counter_offpeak | 42,660 | 79.4% | -15.6pp | 149 |
| aq2 | prevail_onpeak | 613,842 | 95.9% | +0.9pp | 372 |
| aq2 | prevail_offpeak | 571,503 | 96.0% | +1.0pp | 386 |
| aq2 | counter_onpeak | 42,768 | 79.4% | -15.6pp | 160 |
| aq2 | counter_offpeak | 44,193 | 78.2% | -16.8pp | 150 |
| aq3 | prevail_onpeak | 572,352 | 95.7% | +0.7pp | 337 |
| aq3 | prevail_offpeak | 537,288 | 95.7% | +0.7pp | 336 |
| aq3 | counter_onpeak | 37,794 | 76.6% | -18.4pp | 152 |
| aq3 | counter_offpeak | 38,979 | 78.6% | -16.4pp | 141 |
| aq4 | prevail_onpeak | 569,436 | 95.8% | +0.8pp | 334 |
| aq4 | prevail_offpeak | 541,257 | 95.8% | +0.8pp | 343 |
| aq4 | counter_onpeak | 39,225 | 78.0% | -17.0pp | 157 |
| aq4 | counter_offpeak | 43,992 | 80.5% | -14.5pp | 140 |

---

## 3. Direction x Magnitude

Magnitude bins are quartiles of |baseline|. mag_q1=smallest, mag_q4=largest.

### R1

|baseline| quartiles (aq1): p25=50, p50=166, p75=481, p95=1,876

| Quarter | Segment | Count | P95 cov | P95 err | Mean |res| | P95 width |
|---------|---------|------:|--------:|--------:|----------:|----------:|
| aq1 | prevail_mag_q1 | 23,631 | 95.3% | +0.3pp | 225 | 1,848 |
| aq1 | prevail_mag_q4 | 34,466 | 94.4% | -0.6pp | 1,794 | 11,854 |
| aq1 | counter_mag_q1 | 12,157 | 93.8% | -1.2pp | 285 | 1,861 |
| aq1 | counter_mag_q4 | 2,356 | 94.2% | -0.8pp | 2,178 | 11,635 |
| aq4 | prevail_mag_q1 | 22,708 | 95.5% | +0.5pp | 128 | 1,054 |
| aq4 | prevail_mag_q4 | 32,303 | 93.8% | -1.2pp | 1,853 | 10,866 |
| aq4 | counter_mag_q1 | 10,074 | 91.9% | -3.1pp | 187 | 1,044 |
| aq4 | counter_mag_q4 | 1,477 | 96.2% | +1.2pp | 1,870 | 10,751 |

### R2

|baseline| quartiles (aq1): p25=36, p50=120, p75=345, p95=1,277

| Quarter | Segment | Count | P95 cov | P95 err | Mean |res| | P95 width |
|---------|---------|------:|--------:|--------:|----------:|----------:|
| aq1 | prevail_mag_q1 | 193,470 | 96.5% | +1.5pp | 19 | 178 |
| aq1 | prevail_mag_q4 | 265,308 | 93.9% | -1.1pp | 151 | 934 |
| aq1 | counter_mag_q1 | 67,050 | 90.4% | -4.6pp | 40 | 169 |
| aq1 | counter_mag_q4 | 648 | 1.4% | -93.6pp | 653 | 590 |
| aq4 | prevail_mag_q1 | 186,279 | 96.4% | +1.4pp | 17 | 166 |
| aq4 | prevail_mag_q4 | 254,262 | 94.0% | -1.0pp | 147 | 892 |
| aq4 | counter_mag_q1 | 63,711 | 91.1% | -3.9pp | 35 | 159 |
| aq4 | counter_mag_q4 | 831 | 6.9% | -88.1pp | 594 | 674 |

### R3

|baseline| quartiles (aq1): p25=38, p50=121, p75=339, p95=1,327

| Quarter | Segment | Count | P95 cov | P95 err | Mean |res| | P95 width |
|---------|---------|------:|--------:|--------:|----------:|----------:|
| aq1 | prevail_mag_q1 | 242,826 | 96.5% | +1.5pp | 16 | 156 |
| aq1 | prevail_mag_q4 | 313,368 | 93.5% | -1.5pp | 125 | 756 |
| aq1 | counter_mag_q1 | 65,157 | 89.1% | -5.9pp | 37 | 147 |
| aq1 | counter_mag_q4 | 375 | 0.0% | -95.0pp | 571 | 484 |
| aq4 | prevail_mag_q1 | 230,922 | 96.5% | +1.5pp | 15 | 137 |
| aq4 | prevail_mag_q4 | 299,514 | 93.7% | -1.4pp | 113 | 687 |
| aq4 | counter_mag_q1 | 63,117 | 89.2% | -5.8pp | 33 | 131 |
| aq4 | counter_mag_q4 | 387 | 0.0% | -95.0pp | 864 | 564 |

---

## 4. Magnitude Alone

### R1

| Quarter | Mag bin | Count | % | P95 cov | P95 err | Mean |res| | Median |res| | P95 |res| | P95 width |
|---------|---------|------:|--:|--------:|--------:|----------:|----------:|----------:|----------:|
| aq1 | mag_q1 | 36,842 | 25% | 94.9% | -0.1pp | 239 | 90 | 942 | 1,854 |
| aq1 | mag_q2 | 36,811 | 25% | 94.7% | -0.3pp | 426 | 225 | 1,414 | 2,850 |
| aq1 | mag_q3 | 36,836 | 25% | 94.7% | -0.3pp | 709 | 427 | 2,248 | 4,763 |
| aq1 | mag_q4 | 36,826 | 25% | 94.4% | -0.6pp | 1,818 | 1,045 | 6,000 | 11,840 |
| aq2 | mag_q1 | 37,126 | 25% | 95.0% | +0.0pp | 212 | 80 | 811 | 1,643 |
| aq2 | mag_q2 | 37,119 | 25% | 95.0% | -0.0pp | 400 | 220 | 1,338 | 2,724 |
| aq2 | mag_q3 | 37,126 | 25% | 94.8% | -0.2pp | 756 | 470 | 2,412 | 5,065 |
| aq2 | mag_q4 | 37,120 | 25% | 93.8% | -1.2pp | 2,418 | 1,481 | 7,893 | 15,507 |
| aq3 | mag_q1 | 34,188 | 25% | 94.9% | -0.1pp | 183 | 69 | 728 | 1,439 |
| aq3 | mag_q2 | 34,170 | 25% | 94.8% | -0.2pp | 340 | 191 | 1,152 | 2,311 |
| aq3 | mag_q3 | 34,180 | 25% | 95.0% | -0.0pp | 640 | 410 | 1,990 | 4,149 |
| aq3 | mag_q4 | 34,178 | 25% | 94.1% | -0.9pp | 2,025 | 1,368 | 6,093 | 11,956 |
| aq4 | mag_q1 | 33,801 | 25% | 94.5% | -0.5pp | 142 | 58 | 551 | 1,050 |
| aq4 | mag_q2 | 33,764 | 25% | 94.5% | -0.6pp | 277 | 164 | 899 | 1,740 |
| aq4 | mag_q3 | 33,785 | 25% | 94.6% | -0.4pp | 545 | 387 | 1,590 | 3,537 |
| aq4 | mag_q4 | 33,784 | 25% | 93.9% | -1.1pp | 1,854 | 1,240 | 5,841 | 10,861 |

### R2

| Quarter | Mag bin | Count | % | P95 cov | P95 err | Mean |res| | Median |res| | P95 |res| | P95 width |
|---------|---------|------:|--:|--------:|--------:|----------:|----------:|----------:|----------:|
| aq1 | mag_q1 | 265,749 | 25% | 95.0% | +0.0pp | 24 | 12 | 87 | 175 |
| aq1 | mag_q2 | 266,142 | 25% | 94.6% | -0.4pp | 41 | 27 | 125 | 248 |
| aq1 | mag_q3 | 265,962 | 25% | 95.2% | +0.2pp | 63 | 44 | 184 | 380 |
| aq1 | mag_q4 | 265,956 | 25% | 93.7% | -1.3pp | 152 | 94 | 490 | 933 |
| aq2 | mag_q1 | 272,850 | 25% | 95.1% | +0.1pp | 23 | 10 | 84 | 171 |
| aq2 | mag_q2 | 272,955 | 25% | 94.5% | -0.5pp | 42 | 27 | 128 | 250 |
| aq2 | mag_q3 | 272,892 | 25% | 95.0% | +0.0pp | 67 | 46 | 199 | 411 |
| aq2 | mag_q4 | 272,862 | 25% | 94.2% | -0.8pp | 168 | 101 | 521 | 995 |
| aq3 | mag_q1 | 250,932 | 25% | 95.2% | +0.2pp | 21 | 10 | 77 | 158 |
| aq3 | mag_q2 | 251,052 | 25% | 94.5% | -0.5pp | 39 | 24 | 120 | 234 |
| aq3 | mag_q3 | 250,908 | 25% | 95.2% | +0.2pp | 62 | 43 | 180 | 377 |
| aq3 | mag_q4 | 250,938 | 25% | 93.5% | -1.5pp | 146 | 95 | 439 | 841 |
| aq4 | mag_q1 | 255,105 | 25% | 95.1% | +0.1pp | 21 | 10 | 79 | 164 |
| aq4 | mag_q2 | 254,883 | 25% | 94.6% | -0.4pp | 39 | 24 | 125 | 244 |
| aq4 | mag_q3 | 255,180 | 25% | 95.1% | +0.1pp | 63 | 43 | 188 | 384 |
| aq4 | mag_q4 | 255,093 | 25% | 93.7% | -1.3pp | 149 | 94 | 468 | 892 |

### R3

| Quarter | Mag bin | Count | % | P95 cov | P95 err | Mean |res| | Median |res| | P95 |res| | P95 width |
|---------|---------|------:|--:|--------:|--------:|----------:|----------:|----------:|----------:|
| aq1 | mag_q1 | 314,007 | 25% | 95.0% | +0.0pp | 20 | 10 | 75 | 154 |
| aq1 | mag_q2 | 313,587 | 25% | 94.8% | -0.2pp | 34 | 21 | 107 | 213 |
| aq1 | mag_q3 | 313,671 | 25% | 95.1% | +0.1pp | 53 | 35 | 159 | 328 |
| aq1 | mag_q4 | 313,743 | 25% | 93.4% | -1.6pp | 126 | 80 | 399 | 755 |
| aq2 | mag_q1 | 320,136 | 25% | 95.1% | +0.1pp | 19 | 9 | 70 | 142 |
| aq2 | mag_q2 | 319,758 | 25% | 94.8% | -0.2pp | 34 | 22 | 104 | 205 |
| aq2 | mag_q3 | 320,001 | 25% | 95.2% | +0.2pp | 52 | 35 | 156 | 319 |
| aq2 | mag_q4 | 319,947 | 25% | 94.0% | -1.0pp | 135 | 85 | 411 | 782 |
| aq3 | mag_q1 | 297,828 | 25% | 95.1% | +0.1pp | 18 | 8 | 66 | 135 |
| aq3 | mag_q2 | 297,849 | 25% | 94.7% | -0.3pp | 31 | 20 | 96 | 191 |
| aq3 | mag_q3 | 297,873 | 25% | 95.1% | +0.1pp | 48 | 33 | 141 | 293 |
| aq3 | mag_q4 | 297,813 | 25% | 93.2% | -1.8pp | 117 | 74 | 351 | 675 |
| aq4 | mag_q1 | 299,577 | 25% | 95.0% | +0.0pp | 18 | 9 | 67 | 135 |
| aq4 | mag_q2 | 300,222 | 25% | 94.8% | -0.2pp | 31 | 20 | 95 | 189 |
| aq4 | mag_q3 | 299,934 | 25% | 95.3% | +0.3pp | 48 | 34 | 138 | 285 |
| aq4 | mag_q4 | 299,901 | 25% | 93.5% | -1.5pp | 114 | 73 | 357 | 687 |

---

## 5. Key Findings and Implications

### Finding 1: Counter-flow paths are catastrophically under-covered in R2/R3

R2/R3 use MTM (prior-round clearing) as baseline. When a path clears in the same direction
as the prior round (prevail), the residual |mcp - mtm| is small and the band covers well.
When the path reverses (counter), the residual is large — often exceeding the P95 band width
calibrated on the pooled population where prevail paths dominate (~90%).

- R2 counter P95 coverage: **~80%** (target 95%, deficit ~15pp)
- R3 counter P95 coverage: **~78%** (target 95%, deficit ~17pp)
- Counter paths are ~8-9% of R2 and ~7% of R3

### Finding 2: R1 counter under-coverage is milder but still meaningful

R1 uses nodal_f0 which is a model forecast, not a prior clearing. The residual distribution
is already wide for both directions, so the asymmetry is smaller (~2-4pp) but still present.

### Finding 3: Magnitude amplifies the counter problem

Large-magnitude counter paths (|baseline| in top quartile) have the worst coverage because:
1. The baseline is large and in the wrong direction
2. The absolute residual is correspondingly large
3. But the band width was calibrated on a bin dominated by prevail paths

### Finding 4: Class (onpeak/offpeak) is secondary to direction

The per-class stratification in v3 is working well — onpeak/offpeak gaps are <1pp within
each direction segment. Direction is the dominant segmentation axis, not class.

### Implication: Direction-aware calibration needed

Options to explore:
1. **Stratify by direction**: calibrate separate widths for prevail vs counter within each (bin, class)
2. **Asymmetric bands**: widen the band on the counter side (toward baseline sign flip)
3. **Direction as a feature**: add sign(baseline) to the bin scheme
4. **Accept and document**: counter paths are inherently harder to cover; document the limitation

