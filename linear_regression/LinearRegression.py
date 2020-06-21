import pandas as pd
import numpy as np
import sys

class LinearRegression:
	def __init__(self, learning_rate=0.01, regularization_param=0.0, gradient_type='base', learning_rate_factor=1):
		self.learning_rate = learning_rate
		self.regularization_param = regularization_param
		self.gradient_type = gradient_type

		self.w = None
		self.x_mean = None
		self.x_std = None

		self.fit = self.__fit_with_stochastic_gd if gradient_type == 'SGD' else self.__fit_with_gradient_descent

		self.learning_rate_factor = learning_rate_factor

	@staticmethod
	def get_np_array(x):
		if type(x).__module__ != np.__name__:
			return x.values
		else:
			return x

	@staticmethod
	def add_intercept(x):
		x0 = np.ones((x.shape[0], 1))
		return np.c_[x0, x]

	@staticmethod
	def shuffle_dataset(x, y):
		c_data = np.c_[x, y]
		np.random.shuffle(c_data)
		return c_data[:, 0:-1], c_data[:, [-1]]

	@staticmethod
	def get_ridge_gradient_and_error_regulazator(regularization_param, w, n):
		gradient_regularizator = regularization_param*np.sum(w[0, 1:])/n
		error_regularizator = regularization_param * np.sum(w[0, 1:]**2)
		return gradient_regularizator, error_regularizator

	def normalize(self, x):
		return (x - self.x_mean) / self.x_std

	# todo: very naive method, results in more iteration till convergence, find better solution and remove it. source: https://stats.stackexchange.com/questions/46130/determine-the-optimum-learning-rate-for-gradient-descent-in-linear-regression
	def adapt_learning_rate_by_factor(self, prior_mse, mse):
		if mse <= prior_mse:
			self.learning_rate = self.learning_rate * self.learning_rate_factor
		else:
			self.learning_rate = self.learning_rate / self.learning_rate_factor

	# todo: not good, need to read well http://www.onmyphd.com/?p=gradient.descent
	def adapt_learning_rate_by_barzilai(self, prior_gradients, gradients, prior_lr, lr):
		delta_g = gradients - prior_gradients
		delta_x = lr - prior_lr
		new_lr = (delta_g * delta_x) / (delta_g**2)
		return new_lr, lr

	def __fit_with_gradient_descent(self, x, y, epochs, log_epoch=1000):
		x = LinearRegression.get_np_array(x)
		y = LinearRegression.get_np_array(y)

		self.x_mean = np.mean(x, axis=0)
		self.x_std = np.std(x, axis=0)
		x = self.normalize(x)
		x = LinearRegression.add_intercept(x)

		n, m = x.shape
		self.w = np.random.random((1, m))
		prior_lr = 0
		gradients = np.zeros((1, m))
		for i in range(epochs):
			predicted = x.dot(self.w.T)
			gradient_regularizator, error_regularizator = LinearRegression.get_ridge_gradient_and_error_regulazator(
				self.regularization_param, self.w, n)
			errors = predicted - y + gradient_regularizator
			prior_gradients = gradients
			gradients = errors.T.dot(x) / n
			self.w = self.w - self.learning_rate * gradients

			mean_squared_error = (errors.T.dot(errors) + error_regularizator) / n
			gradients_norm = abs(gradients).sum()
			# self.learning_rate, prior_lr = self.adapt_learning_rate_by_barzilai(prior_gradients, gradients, prior_lr, self.learning_rate)

			if i % log_epoch == 0:
				print(f"Iter: {i}, gradients norm: {gradients_norm}, mse: {mean_squared_error}")

	def __fit_with_stochastic_gd(self, x, y, epochs, log_epoch=1000):
		x = LinearRegression.get_np_array(x)
		y = LinearRegression.get_np_array(y)

		self.x_mean = np.mean(x, axis=0)
		self.x_std = np.std(x, axis=0)
		x = self.normalize(x)
		x = LinearRegression.add_intercept(x)

		x, y = LinearRegression.shuffle_dataset(x, y)

		n, m = x.shape
		self.w = np.random.random((1, m))
		for i in range(epochs):
			for row_index in range(n):
				gradient_regularizator, error_regularizator = LinearRegression.get_ridge_gradient_and_error_regulazator(
					self.regularization_param, self.w, n)
				x_row = x[row_index, :].reshape(1, -1)
				pred = x_row.dot(self.w.T)
				errors = y[row_index] - pred + gradient_regularizator
				gradients = errors.dot(-2).T.dot(x_row)
				self.w = self.w - self.learning_rate*gradients

				mean_square_error = (errors.T.dot(errors) + error_regularizator)/n
				gradients_norm = abs(gradients).sum()
				if i % log_epoch == 0 and row_index % 100 == 0:
					print(f"Iter: {i}, row: {row_index}, gradients norm: {gradients_norm}, mse: {mean_square_error}")

	def predict(self, x):
		x = LinearRegression.get_np_array(x)
		x = self.normalize(x)
		x = self.add_intercept(x)
		return x.dot(self.w.T)


if __name__ == "__main__":
	pd.set_option('display.max_rows', 500)
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.random.seed(1)
	np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})

	# data = pd.read_csv('../data/house.csv')
	data = pd.read_csv('../data/boston.csv')
	y = data.iloc[:, [-1]]
	x = data.iloc[:, 0:-1]

	model_lr = LinearRegression(0.01, regularization_param=1e-2, gradient_type='batch')
	# model_lr = LinearRegression(0.001, regularization_param=1e-3, gradient_type='SGD')
	model_lr.fit(x, y, epochs=int(1e4), log_epoch=100)
	prediction = model_lr.predict(x)
	data['predicted'] = prediction
	print(data)
	print(f"MSE: {((data.iloc[:, -2] - data.iloc[:, -1])**2).sum() / data.shape[0]}")

	# data_new = pd.read_csv('../data/house_new.csv')
	# data_new['predicted'] = model_lr.predict(data_new)
	# print(data_new)


