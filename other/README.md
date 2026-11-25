# Other material

Code that is not needed to reproduce the paper, kept for reference. It is not maintained: paths and imports refer to the layout at the time of the experiments (for example `scripts/` and `src/models/third_party_models/`) and may need adjustment.

| Path | Content |
| --- | --- |
| `emu3_legacy.py` | the detector as used for the experiments, including an unused beam-search variant and attention-map export |
| `data/mapillary.py` | Mapillary Vistas loader (not used in the paper) |
| `scripts/sweep.sh` | generic Hydra sweep over SNR, temperature, window size, LMM on/off and LDPC on/off |
| `scripts/sweep_*_fast.py` | sweeps that load the models once |
| `scripts/sweep_missing_*.sh`, `scripts/sweep_append_ldpctrue.sh`, `scripts/sweep_transition_*.sh` | one-off runs that completed the paper sweeps |
| `scripts/compare_ldpc.py`, `scripts/plot_*.py` | auxiliary plots and tables |
