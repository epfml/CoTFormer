from . import base
from . import cotformer_full_depth
from . import but_halting_freeze_input_on_stop
from . import but_mod_efficient_sigmoid_lnmid_depthemb_random_factor
from . import cotformer_full_depth_lnmid_depthemb
from . import but_full_depth
from . import adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final
from . import pondernet
from . import but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute

MODELS = {
    "base": base.GPTBase,
    "adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final": adaptive_cotformer_mod_efficient_sigmoid_crw_lnmid_de_random_factor_single_final.GPTBase,
    "but_full_depth": but_full_depth.GPTBase,
    "cotformer_full_depth": cotformer_full_depth.GPTBase,
    "cotformer_full_depth_lnmid_depthemb": cotformer_full_depth_lnmid_depthemb.GPTBase,
    "but_mod_efficient_sigmoid_lnmid_depthemb_random_factor": but_mod_efficient_sigmoid_lnmid_depthemb_random_factor.GPTBase,
    "but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute": but_mod_efficient_sigmoid_lnmid_depthemb_random_factor_for_mac_compute.GPTBase,
    "pondernet": pondernet.GPTBase,
    "but_halting_freeze_input_on_stop": but_halting_freeze_input_on_stop.GPTBase,
}


def make_model_from_args(args):
    return MODELS[args.model](args)


def registered_models():
    return MODELS.keys()
