import os
import subprocess

print(os.getcwd())

dirs = [
    './src/sql',
    './plot'
7c473af29d1ebc659c27184e605a768283523
]
for d in dirs:
  os.mkdir(d)

subprocess.call(['git', 'init'])
subprocess.call(['git', 'submodule', 'add', 'https://github.com/MasatoTakahashi/myDSUtils'])
subprocess.call(['git', 'flow', 'init'])

