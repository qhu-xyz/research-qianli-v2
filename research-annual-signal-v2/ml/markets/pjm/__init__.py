"""ml.markets.pjm — PJM-specific annual signal implementation.

PJM annual has:
  - 4 rounds (R1-R4), all in April
  - Different class types: onpeak, dailyoffpeak, wkndonpeak (NO offpeak)
  - Different bridge/CID mapping than MISO
  - Different DA shadow price source

Current status: skeleton only. No implementation yet.
Depends on: ml.core contracts being stable before PJM-specific code is written.
"""
