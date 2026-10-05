import numpy as np

d = np.load("clear.npz")

print("Unique buildings:", len(np.unique(d["building_id_code"])))
print("Roof faces:", np.sum(d["is_roof"]))
print("Wall faces:", np.sum(d["is_wall"]))

roof = d["annual_irradiance_wh_m2"][d["is_roof"]]

print("Roof irradiance percentiles:")
print(np.nanpercentile(roof, [0, 5, 25, 50, 75, 95, 100]))