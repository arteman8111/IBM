import matplotlib.pyplot as plt
import numpy as np

def get_plot(t,param,label): 

    plt.figure(dpi=100)
    plt.plot(t,param)
    
    # Adding details to the plot
    plt.title('Расчёт параметров')
    plt.xlabel('t,[c]')
    plt.ylabel(label)

    # Displaying the plot
    plt.show()