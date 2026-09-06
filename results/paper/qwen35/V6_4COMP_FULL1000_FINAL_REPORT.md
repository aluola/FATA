================================================================
Q-FATA V6 4-COMPRESSOR FULL1000 REPORT
================================================================
configs: ['v6_a_fullpres_light_4comp'] | methods: ['VisionZIP', 'VisPruner', 'FlowCut', 'PruMerge'] | scorer: canonical strict VQA

Full       Clean=  89.55  FATA=  88.92  Damage=  +0.62 pp  (n=16000)
1/3        Clean=  80.66  FATA=  72.47  Damage=  +8.19 pp  (n=16000)
Practical  Clean=  65.10  FATA=  48.48  Damage= +16.62 pp  (n=16000)

Amp 1/3       = +7.56 pp
Amp Practical = +15.99 pp

Transitions: Full CW/WC=232/132 NetHarm=100
             1/3  CW/WC=1729/419 NetHarm=1310
             Practical CW/WC=3250/591 NetHarm=2659
FullPreservation=0.984  CBR_conditional=0.328

Validity: NaN=0 Inf=0 errors=0 invalid_delta=0 shared_delta_violations=0 maxLinf=0.007843

Diagnostics (NON-GATING): datasets AmpPrac>0 = 4/4, method x dataset AmpPrac>0 = 16/16

Per dataset (DIAGNOSTIC):
  TextVQA_Open   Dfull= +0.70 D13= +5.22 Amp13= +4.52 Dprac= +8.60 AmpPrac= +7.90
  VQAv2_Open     Dfull= +1.00 D13=+10.00 Amp13= +9.00 Dprac=+14.57 AmpPrac=+13.57
  ScienceQA_MC   Dfull= -0.20 D13= +9.47 Amp13= +9.67 Dprac=+23.70 AmpPrac=+23.90
  VQAv2_MC       Dfull= +1.00 D13= +8.05 Amp13= +7.05 Dprac=+19.60 AmpPrac=+18.60

================================================================
FOUR-COMP MACRO GATE
================================================================
Full Damage <= 3 pp:       PASS  (+0.62)
1/3 Damage >= 8 pp:        PASS  (+8.19)
Practical Damage >= 10 pp: PASS  (+16.62)
Amp 1/3 >= 6 pp:           PASS  (+7.56)
Amp Practical >= 8 pp:     PASS  (+15.99)
Practical NetHarm > 0:     PASS  (2659)
Validity:                  PASS

FOUR_COMP_GATE_PASS = YES