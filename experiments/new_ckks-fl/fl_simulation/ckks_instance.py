from ckks.main import CKKS
from fl_simulation.model.model import Net, get_weights
from utils.flatten import get_structure
from utils.numbers import next_power_of_two

_template_weights = get_weights(Net())
MODEL_STRUCTURE = get_structure(_template_weights)
MODEL_SIZE = int(sum(weight.size for weight in _template_weights))
SLOT_COUNT = next_power_of_two(max(MODEL_SIZE, 2**5))

ckks: CKKS = CKKS(
    N=SLOT_COUNT,
    sigma=3,
    model_size=MODEL_SIZE,
    fix_a=True,
    structure=MODEL_STRUCTURE,
)

