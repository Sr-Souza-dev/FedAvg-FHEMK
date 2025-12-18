from ckks.main import CKKS
# from fl_simulation.model.model import get_weights, Net
# from utils.flatten import flatten
# from utils.numbers import next_power_of_two

# weights = get_weights(Net())
# flat_weights = flatten(weights)

ckks:CKKS = CKKS(
    N=2**5,
    sigma=3,
    model_size=44426,
    fix_a=True
)

