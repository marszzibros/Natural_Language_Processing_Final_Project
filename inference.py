from utils import Model
import os
import sys
import pandas as pd

model_id = sys.argv[1]
city = sys.argv[2]
mode = sys.argv[3]

llm_agent = Model(model_id=model_id)

image_paths = []

folder_path = f"/gpfs2/scratch/xzhang31/VIGOR/{city}/{mode}"

df = pd.read_csv(f"data/{city}.csv")

for file_path in df[mode]:
	llm_agent.generate(image_path=os.path.join(folder_path, file_path), image_type=mode)

