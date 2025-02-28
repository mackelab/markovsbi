

## Illustrative + comparative proposal construction (Fig 5)

Illustrative plots in "lotka_volterra_proposal_plots.ipynb". Comparative plots require the following experiments to be run:

```bash
markovsbi +experiment=bm_lv_score100k
markovsbi +experiment=bm_lv_score100k_good_proposal
```

Panel produced in "additional_analysis.ipynb".



### Appendix plot by num simulations (Fig 6)

Needs following experiments to be run (some might already be run e.g. SIR/LV):
```bash
markovsbi +experiment=bm_nle1000k ;
markovsbi +experiment=bm_nre1000k ;
markovsbi +experiment=bm_npe1000k ;
markovsbi +experiment=bm_score1000k ;
markovsbi +experiment=bm_lv_nle10k ;
markovsbi +experiment=bm_lv_nle100k ;
markovsbi +experiment=bm_lv_nle1000k ;
markovsbi +experiment=bm_lv_nre10k ;
markovsbi +experiment=bm_lv_nre100k ;
markovsbi +experiment=bm_lv_nre1000k ;
markovsbi +experiment=bm_lv_npe10k ;
markovsbi +experiment=bm_lv_npe100k ;
markovsbi +experiment=bm_lv_npe1000k ;
markovsbi +experiment=bm_lv_score10k ;
markovsbi +experiment=bm_lv_score100k ;
markovsbi +experiment=bm_lv_score1000k ;
markovsbi +experiment=bm_sir_nle10k ;
markovsbi +experiment=bm_sir_nle100k ;
markovsbi +experiment=bm_sir_nle1000k ;
markovsbi +experiment=bm_sir_nre10k ;
markovsbi +experiment=bm_sir_nre100k ;
markovsbi +experiment=bm_sir_nre1000k ;
markovsbi +experiment=bm_sir_npe10k ;
markovsbi +experiment=bm_sir_npe100k ;
markovsbi +experiment=bm_sir_npe1000k ;
markovsbi +experiment=bm_sir_score10k ;
markovsbi +experiment=bm_sir_score100k ;
markovsbi +experiment=bm_sir_score1000k ;
```

Figure produced from data in "additional_analysis.ipynb".

## Additional calibration analysis of LV/SIR (Fig. 7)

This needs the following experiments to be run:

```bash
markovsbi +experiment=bm_lv_score100k ;
markovsbi +experiment=bm_sir_score100k ;
markovsbi +experiment=bm_lv_npe100k ;
markovsbi +experiment=bm_sir_npe100k ;
```

The figures are then predouced in "lotka_volterra_calibration.ipynb" and "sir_calibration.ipynb" and combined in "fig_aditional.ipynb".

# Additional baselines (Tab. 3, Tab 4.)

This will require an bunch of additional experiments to be run. The following experiments need to be run:

```bash
markovsbi +experiment=bm_npe_sstat10k;
markovsbi +experiment=bm_npe_sstat100k;
markovsbi +experiment=bm_npe_sstat10k_alt_embedding;
markovsbi +experiment=bm_npe_sstat100k_alt_embedding;
markovsbi +experiment=bm_npe_sliced_sstat10k;
markovsbi +experiment=bm_npe_sliced_sstat100k;
markovsbi +experiment=bm_npe_sliced_sstat10k_alt_embedding;
markovsbi +experiment=bm_npe_sliced_sstat100k_alt_embedding;
markovsbi +experiment=bm_npe10k_alt_embedding;
markovsbi +experiment=bm_npe100k_alt_embedding;
markovsbi +experiment=bm_npe10k_long_sims;
markovsbi +experiment=bm_npe100k_long_sims;
markovsbi +experiment=bm_nle_sstat10k;
markovsbi +experiment=bm_nle_sstat100k;
markovsbi +experiment=bm_nle_sliced_sstat10k;
markovsbi +experiment=bm_nle_sliced_sstat100k;
markovsbi +experiment=bm_nre_sstat10k;
markovsbi +experiment=bm_nre_sstat100k;
markovsbi +experiment=bm_nre_sliced_sstat10k;
markovsbi +experiment=bm_nre_sliced_sstat100k;
markovsbi +experiment=bm_nse10k;
markovsbi +experiment=bm_nse100k;
```

The same set of experiments needs also be run for LV and SIR additionally.

## Additional results on proposal (Tab 1., Tab 2.)

This will require the following experiments to be run:

```bash
markovsbi +experiment=bm_score100k_pred_proposal
markovsbi +experiment=bm_sir_score100k_good_proposal
markovsbi +experiment=bm_lv_score100k_good_proposal
```
