import os
import sys

rootdir = sys.argv[1]
taskdir = sys.argv[2]

absrootdir = os.path.abspath(rootdir)
abstaskdir = os.path.abspath(taskdir)

for root, subfolders, files in os.walk(absrootdir):
	for fold in subfolders:
		if not os.path.exists(os.path.join(abstaskdir, fold)):
			os.makedirs(os.path.join(abstaskdir, fold))