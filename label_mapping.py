"""Post-prediction label remap, applied AFTER the model's own prediction
-- the model itself never changes for this failure mode. Identity by
default. To simulate a class/label-mapping bug (e.g. a mixed-up output
encoding shipped in a deploy config), edit LABEL_MAP so keys don't map
to themselves.

Since confidence/entropy are computed from the model's raw probabilities
BEFORE this remap is applied, a wrong mapping produces a very specific,
different failure signature than a bad checkpoint: accuracy craters
while confidence stays exactly as high as before -- the model is just as
"sure" of itself, only the label it's sure of got relabeled wrong.
"""
LABEL_MAP = {0: 0, 1: 1}
