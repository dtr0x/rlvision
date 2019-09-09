import os

for i in range(4, 250):
	model_file = "models/target_net_{}.pt".format(i)
	os.rename(model_file, "plane_models/target_net_{}.pt".format(i))