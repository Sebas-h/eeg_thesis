# from sacred_init import ex

# x_choices = [2, 3, 4]
# for x in x_choices:
#     ex.run(config_updates={'x': x})

from sacred_experiment import ex

# for x in range(2):
#     ex.run(named_configs=['variant' + str(x + 1)])

r = ex.run()
rr = ex.run(named_configs=['variant1'])
