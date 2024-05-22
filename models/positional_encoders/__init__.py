from . import encoder, rotary

POS_ENCS = {
    "none": encoder.PositionalEncoder,
    "rotary": rotary.RotaryPositionalEncoder,
}


def get_encoder(encoder_name):
    return POS_ENCS[encoder_name]


def registered_encoders():
    return POS_ENCS.keys()
