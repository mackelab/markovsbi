## Figure 2 (and Fig 8., Fig 9., Fig 10., Fig 11., Fig. 13)

To produce the main results of Figure 2 you will have to run the following experiments:
```bash
markovsbi +experiment=baseline_npe ;
markovsbi +experiment=baseline_nre &
markovsbi +experiment=baseline_score;
markovsbi +experiment=baseline_nle ;
markovsbi +experiment=bm_nle10k ;
markovsbi +experiment=bm_nre10k ;
markovsbi +experiment=bm_npe10k ;
markovsbi +experiment=bm_score10k ;
markovsbi +experiment=bm_nle100k ;
markovsbi +experiment=bm_nre100k ;
markovsbi +experiment=bm_npe100k ;
markovsbi +experiment=bm_score100k ;
markovsbi +experiment=bm_proposal ;
markovsbi +experiment=bm_proposal_mixture;
markovsbi +experiment=bm_sampler100k ;
markovsbi +experiment=bm_sampler100k_T100 ;
markovsbi +experiment=bm_sampler100k_eval ;
markovsbi +experiment=bm_sampler100k_T100_eval ;
```

Subpanels are constructed in `main_panels.ipynb` and combined in `fig2.ipynb` (not only fig2 but also appendix figures).


## Appendix and more

### Partially factorized method ablation

Run the following experiments to reproduce the results of the partially factorized methods.
```bash
markovsbi +experiment=bm5_nle10k ;
markovsbi +experiment=bm5_nre10k ;
markovsbi +experiment=bm5_npe10k ;
markovsbi +experiment=bm5_score10k ;
markovsbi +experiment=bm5_nle100k ;
markovsbi +experiment=bm5_nre100k ;
markovsbi +experiment=bm5_npe100k ;
markovsbi +experiment=bm5_score100k ;
```

Subpannels are produced in "benchmark_pfn5.ipynb" and combined in "fig2.ipynb".

### Gaussina proposal ablation

Run the following experiments to reproduce the results of the Gaussian proposal ablation.
```bash
markovsbi +experiment=bm_proposal ;
markovsbi +experiment=bm_proposal_mixture;
```

Subpannels are produced in "benchmark_proposal.ipynb" and combined in "fig2.ipynb".

### Sampler ablation

This would require the following experiments to be run:
```bash
markovsbi +experiment=bm_sampler100k ;
markovsbi +experiment=bm_sampler100k_T100 ;
markovsbi +experiment=bm_sampler100k_eval ;
markovsbi +experiment=bm_sampler100k_T100_eval ;
```

Subpannels are produced in "main_panels.ipynb" and combined in "fig2.ipynb".
