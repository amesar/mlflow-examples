import numpy as np
import matplotlib.pyplot as plt

def create_plot_file(y_test_set, y_predicted, plot_file):
    global image
    fig, ax = plt.subplots()
    ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")
    #plt.show()

    image = fig
    fig.savefig(plot_file)
    plt.close(fig)
