import sys, time, os 
import shutil
import subprocess

if __name__=='__main__':
  dirs = ['../eda','../input', '../submit', '../intermediate']
  for d in dirs:
    os.mkdir(d)
  
  subprocess.call(['git', 'init'])  
  subprocess.call(['git', 'flow', 'init'])
