import os
import os.path as osp
import matplotlib.pyplot as plt
from sys import platform

def plot_train_val_losses(hist, show=False, save=True, filename='Train_Val_HIST.png'):
	"""Loss tracker
	
	Plot the losses of the network to see the trend

	Arguments:
		hist {[dict]} -- Tracking variables

	Keyword Arguments:
		show {bool} -- If to display the figure (default: {False})
		save {bool -- If to store the figure (default: {True})
		filename {str} -- filename to assign to the figure (default: {'Train_Val_HIST.png'})
	"""

	x = range(len(hist['train_losses']))
	
	y1 = hist['train_losses']
	y2 = hist['val_losses']

	plt.plot(x, y1, label='Training Loss')
	plt.plot(x, y2, label='Validation Loss')

	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.legend(loc=2)
	plt.tight_layout()

	if platform == 'linux' or platform == 'linux2':
		save_dir = '/data/home/apw858/python_scripts/machine_learning/geometric/results'
	elif platform == 'win32':
		save_dir = 'C:\\Users\\marco\\Documents\\PhD\\Codes\\Results'

	if not osp.exists(save_dir):
		os.makedirs(save_dir)

	PATH = osp.join(save_dir, filename)

	if save:
		plt.savefig(PATH)

	if show:
		plt.show()
	else:
		plt.close()


def plot_predictions_to_real(hist):
	"""Loss tracker

	Plot the losses of the network to see the trend

	Arguments:
		hist {[dict]} -- Tracking variables

	Keyword Arguments:
		show {bool} -- If to display the figure (default: {False})
		save {bool -- If to store the figure (default: {True})
		filename {str} -- filename to assign to the figure (default: {'Train_Val_HIST.png'})
	"""

	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=[13.3, 10])
	fig.suptitle('Predictions vs Real Values')

	x1 = hist['b_reals']
	y1 = hist['b_pred']

	x2 = hist['l_reals']
	y2 = hist['l_pred']

	x3 = hist['h_reals']
	y3 = hist['h_pred']

	ax1.scatter(x1, y1)
	ax1.set_xlabel('Real Values')
	ax1.set_ylabel('Predicted Values')
	ax1.set_title('Bandgap')

	ax2.scatter(x2, y2)
	ax2.set_xlabel('Real Values')
	ax2.set_ylabel('Predicted Values')
	ax2.set_title('Lumo')

	ax3.scatter(x3, y3)
	ax3.set_xlabel('Real Values')
	ax3.set_ylabel('Predicted Values')
	ax3.set_title('Homo')

	plt.savefig('results.png')

	return fig
