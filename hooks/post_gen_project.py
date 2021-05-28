import os
import subprocess

print(os.getcwd())

dirs = [
    './input',
    './submit',
    './src/sql',
    './intermediate',
    './intermediate/proc_data',
    './plot',
    './model',
    './log'
]
for d in dirs:
  os.mkdir(d)

subprocess.call(['git', 'init'])
subprocess.call(['git', 'submodule', 'add', 'https://github.com/MasatoTakahashi/myDSUtils'])
subprocess.call(['git', 'flow', 'init'])

