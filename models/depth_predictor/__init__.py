from . import sb_act

PREDICTORS = {
    "sb_act": sb_act.StickBreakingACTPredictor,
}


def get_depth_predictor(name):
    return PREDICTORS[name]


def registered_predictors():
    return PREDICTORS.keys()
