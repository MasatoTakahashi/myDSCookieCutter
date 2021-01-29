import sys, time, os 
import shutil
import subprocess

if __name__=='__main__':
  dirs = ['../input', '../submit', '../intermediate']
  for d in dirs:
    os.mkdir(d)
  
  subprocess.call(['git', 'init'])
  subprocess.call(['git', 'submodule', 'add', 'https://github.com/MasatoTakahashi/myDSUtils'])
  subprocess.call(['git', 'flow', 'init'])
