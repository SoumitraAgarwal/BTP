import os
import shutil


shutil.rmtree('Data/')
shutil.rmtree('Model/')
shutil.rmtree('Plots/')
os.mkdir('Data')
os.mkdir('Model')
os.mkdir('Plots')
open('Data/0.csv', 'wb')
shutil.copy2('data.csv', 'Data/0.csv')
