import os

for i in range(0, 496):
	model_file = "plane_models/target_net_{}.pt".format(i)
	os.rename(model_file, "plane_models/target_net_{}.pt".format(i+4))