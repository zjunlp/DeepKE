import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


name_list = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '35-42']

data1 = [70.3, 65.0, 64.0, 63.8, 62.7, 59.2, 57.7, 55.3 ]
data2 = [70.0, 64.2, 63.1, 62.0, 60.5, 56.1, 54.3, 51.4 ]

x =list(range(len(data1)))


plt.figure(figsize=(6, 3))
plt.plot(x,data1,marker='>',label='w/ U-shaped Network')
plt.plot(x,data2,'--',marker='o',label='w/o U-shaped Network')


plt.ylim([50,75])
plt.xlabel('# of Entities per Document')
plt.ylabel('dev F1 (in %)')
plt.xticks(x,name_list)
plt.legend(loc='upper right')
plt.show()