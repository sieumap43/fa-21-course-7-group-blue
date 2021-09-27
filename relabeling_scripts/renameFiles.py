import numpy as np
from pathlib import Path
import pandas as pd
import uuid
from shutil import copyfile

src = input("Select folder where training data is stored (relative path. Output data will be stored in [SRC]/out): ")

srcPath = Path(src)

# create output directories
destLeft = srcPath / 'out/left'
destCenter = srcPath / 'out/center'
destRight = srcPath / 'out/right'

destLeft.mkdir(parents=True, exist_ok=True)
destCenter.mkdir(parents=True, exist_ok=True)
destRight.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(srcPath / 'driving_log.csv', delimiter=',', header=None, names=['center', 'left', 'right', 'angle', 'acc', 'break', 'speed'], dtype={'center': str, 'left': str, 'right': str, 'angle': float, 'acc': float, 'break': float, 'speed': float})
copyCount = 0
for idx, row in df.iterrows():
	# create new filename: <UUID>_<coded_speed>_<coded_angle>.jpg
	uuidVal = uuid.uuid1() # Generate a UUID from a host ID, sequence number, and the current time.
	codedSpeed = int(row['speed'] * 1000)
	codedAngle = int(row['angle'] * 1000)
	fname = f'{idx}_{codedSpeed}_{codedAngle}.jpg'
	#print(fname)

	srcLeft = Path(row['left'])
	srcCenter = Path(row['center'])
	srcRight = Path(row['right'])

	# Try to copy source files into destination directories, re-labeled
	try:
		copyfile(srcLeft, destLeft / fname)
		copyfile(srcCenter, destCenter / fname)
		copyfile(srcRight, destRight / fname)
		copyCount += 1
	except Exception as e:
		continue

print(f'Successfully copied {copyCount}/{len(df)} frames.')
