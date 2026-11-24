"""Per-channel intensity statistics of the geminin_drugs dataset.

Generated from src/dataset/geminin_drugs_stats.json (written by examine_dataset.py) - do not edit by hand.

The statistics were computed on the TRAIN split AFTER the transforms
"arcsinh butterworth" (examine_dataset.py defaults: arcsinh_cofactor=5.0,
butterworth_cutoff=0.2, butterworth_order=2, per_frame_butterworth=True), so the
"global_norm" / "experiment_norm" transforms are only meaningful when applied after
exactly that preprocessing.

EXPERIMENT_CHANNEL_STATS is keyed by experiment name, i.e. the last component of
metadata["Path"]. Channels that were not recorded in an experiment are omitted.
"""

GLOBAL_CHANNEL_STATS = {
    "Ch_AKT-KTR": {"mean": 6.344773292541504, "std": 0.5240649580955505},
    "Ch_ERK-KTR": {"mean": 6.611674785614014, "std": 1.006008267402649},
    "Ch_Geminin": {"mean": 4.563348293304443, "std": 0.6853934526443481},
    "Ch_H2B": {"mean": 5.41298770904541, "std": 0.8275633454322815},
}

EXPERIMENT_CHANNEL_STATS = {
    "2019-02-22_Exp1_Ctx-Gef-Sor": {
        "Ch_ERK-KTR": {"mean": 6.083251476287842, "std": 0.9049853086471558},
        "Ch_Geminin": {"mean": 4.178114891052246, "std": 0.32053521275520325},
        "Ch_H2B": {"mean": 6.000890731811523, "std": 0.289654403924942},
    },
    "2019-02-24_Exp2_Tra-Shc-Pic": {
        "Ch_ERK-KTR": {"mean": 6.266994476318359, "std": 0.7113966941833496},
        "Ch_Geminin": {"mean": 4.173145294189453, "std": 0.28485795855522156},
        "Ch_H2B": {"mean": 6.019253730773926, "std": 0.25445085763931274},
    },
    "2019-02-26_Exp3_Azd53-Bez-Azd85": {
        "Ch_ERK-KTR": {"mean": 6.151448726654053, "std": 0.726502001285553},
        "Ch_Geminin": {"mean": 4.200264930725098, "std": 0.30265629291534424},
        "Ch_H2B": {"mean": 5.984652042388916, "std": 0.21686188876628876},
    },
    "2019-03-08_Exp1_Ctx-Gef-Sor": {
        "Ch_ERK-KTR": {"mean": 6.058756351470947, "std": 0.9668436646461488},
        "Ch_Geminin": {"mean": 4.114328384399414, "std": 0.2583671510219574},
        "Ch_H2B": {"mean": 5.940809726715088, "std": 0.22556298971176147},
    },
    "2019-03-10_Exp2_Tra-Shc-Pic": {
        "Ch_ERK-KTR": {"mean": 6.26877498626709, "std": 0.5556083917617798},
        "Ch_Geminin": {"mean": 4.121275424957275, "std": 0.2499689757823944},
        "Ch_H2B": {"mean": 6.008088111877441, "std": 0.19427357614040375},
    },
    "2019-03-23_Exp3_Azd53-Bez-Azd85": {
        "Ch_ERK-KTR": {"mean": 7.092031002044678, "std": 0.7571106553077698},
        "Ch_Geminin": {"mean": 4.894405841827393, "std": 0.47362932562828064},
        "Ch_H2B": {"mean": 5.689037799835205, "std": 0.8008717894554138},
    },
    "2019-04-12_Exp1_Ctx-Gef-Sor": {
        "Ch_ERK-KTR": {"mean": 6.919556140899658, "std": 1.0127216577529907},
        "Ch_Geminin": {"mean": 4.217022895812988, "std": 0.33062317967414856},
        "Ch_H2B": {"mean": 4.6413750648498535, "std": 0.6562091112136841},
    },
    "2019-04-26_Exp2_Tra-Shc-Pic": {
        "Ch_ERK-KTR": {"mean": 7.289513111114502, "std": 0.9474064707756042},
        "Ch_Geminin": {"mean": 4.709488868713379, "std": 0.4508257508277893},
        "Ch_H2B": {"mean": 5.328733444213867, "std": 0.7730588316917419},
    },
    "2019-05-03_Exp3_Azd53-Bez-Azd85": {
        "Ch_ERK-KTR": {"mean": 7.6135334968566895, "std": 0.672363817691803},
        "Ch_Geminin": {"mean": 5.245548725128174, "std": 0.7621670961380005},
        "Ch_H2B": {"mean": 5.488863945007324, "std": 0.81059730052948},
    },
    "2019-05-18_Exp1_Ctx-Gef-Sor_NoGem_FoXO": {
        "Ch_AKT-KTR": {"mean": 6.463828086853027, "std": 0.5044090747833252},
        "Ch_ERK-KTR": {"mean": 7.088240146636963, "std": 1.1736184358596802},
        "Ch_H2B": {"mean": 5.224552631378174, "std": 0.8343142867088318},
    },
    "2019-05-26_Exp2_Tra-Shc-Pic_NoGemFoXO": {
        "Ch_AKT-KTR": {"mean": 6.3267292976379395, "std": 0.5006099343299866},
        "Ch_ERK-KTR": {"mean": 6.613216876983643, "std": 1.203251600265503},
        "Ch_H2B": {"mean": 5.091115474700928, "std": 0.8387901782989502},
    },
    "2019-05-31_Exp3_Azd53-Bez-Azd85_NoGemFoXO": {
        "Ch_AKT-KTR": {"mean": 6.127506732940674, "std": 0.4727441072463989},
        "Ch_ERK-KTR": {"mean": 6.565446853637695, "std": 1.1578783988952637},
        "Ch_H2B": {"mean": 4.900542736053467, "std": 0.7501457929611206},
    },
    "2019-06-07_Exp2_Tra-Shc-Pic_NoGemFoXO": {
        "Ch_AKT-KTR": {"mean": 6.479044437408447, "std": 0.5603815913200378},
        "Ch_ERK-KTR": {"mean": 6.895257472991943, "std": 1.190009593963623},
        "Ch_H2B": {"mean": 5.205883502960205, "std": 0.8704143166542053},
    },
    "2020-07-06_E545KandH1047R_Geminin_ERK_drugs": {
        "Ch_ERK-KTR": {"mean": 6.6913743019104, "std": 0.635260820388794},
        "Ch_Geminin": {"mean": 5.444513320922852, "std": 0.41304370760917664},
        "Ch_H2B": {"mean": 4.314070224761963, "std": 0.4112151265144348},
    },
    "2020-07-10_E545KandH1047R_Geminin_ERK_drugs": {
        "Ch_ERK-KTR": {"mean": 6.931107521057129, "std": 0.6609882712364197},
        "Ch_Geminin": {"mean": 5.627851963043213, "std": 0.5442133545875549},
        "Ch_H2B": {"mean": 4.4195637702941895, "std": 0.4901500940322876},
    },
}
