import os
import pandas as pd

os.system("mkdir data")

for city in ["NewYork", "SanFrancisco", "Chicago", "Seattle"]:

    satellite = []
    panorama = []

    with open(f"/gpfs2/scratch/xzhang31/VIGOR/splits/{city}/same_area_balanced_test.txt", "r") as f:
        for line in f:
            line = line.strip().split(" ")
            panorama.append(line[0])
            satellite.append(line[1])

    pd.DataFrame({"satellite": satellite[:2500], "panorama": panorama[:2500]}).to_csv(f"data/{city}.csv", index=False)
            