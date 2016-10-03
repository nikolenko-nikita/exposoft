import os
import sys

rootdir = sys.argv[1]
absrootdir = os.path.abspath(rootdir)

with open('output_files_list.txt','w') as fout:
	for root, subfolders, files in os.walk(absrootdir):
		for f in files:
			fout.write(os.path.join(root, f) + '\n')