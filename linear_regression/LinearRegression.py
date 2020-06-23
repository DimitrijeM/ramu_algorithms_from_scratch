import pandas as pd
import numpy as np
import sys

class LinearRegression:
	def __init__(self, learning_rate=0.01, regularization_param=0.0, gradient_type='base'):
		self.learning_rate = learning_rate
		self.regularization_param = regularization_param
		self.gradient_type = gradient_type

		self.w = None
		self.x_mean = None
		self.x_std = None

		self.fit = self.__fit_with_stochastic_gd if gradient_type == 'SGD' else self.__fit_with_gradient_descent


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

	@staticmethod
	def calculate_mean_squared_error(errors, error_regularizator=0):
		return (errors.T.dot(errors) + error_regularizator) / errors.shape[0]

	@staticmethod
	def predict_and_evaluate(x, y, w):
		predicted = x.dot(w.T)
		errors = predicted - y
		mse = LinearRegression.calculate_mean_squared_error(errors, 0)
		return mse

	@staticmethod
	def adapt_learning_rate_const(learning_rate):
		return learning_rate

	# todo: very naive method, results in more iteration till convergence, find better solution and remove it.
	#  source: https://stats.stackexchange.com/questions/46130/determine-the-optimum-learning-rate-for-gradient-descent-in-linear-regression
	@staticmethod
	def adapt_learning_rate_by_factor(x, y, w, prior_w, learning_rate, learning_rate_factor):
		prior_mse = LinearRegression.predict_and_evaluate(x, y, prior_w)
		mse = LinearRegression.predict_and_evaluate(x, y, w)
		if mse <= prior_mse:
			return learning_rate * learning_rate_factor
		else:
			return learning_rate / learning_rate_factor

	@staticmethod
	def adapt_learning_rate_by_cauchy(x, y, w, gradients, possible_lr=[0.001, 0.005, 0.1, 0.2, 0.3, 0.5]):
		values = np.array([LinearRegression.predict_and_evaluate(x, y, w-lr*gradients) for lr in possible_lr])
		selected_lr = possible_lr[np.argmin(values)]
		# print(possible_lr)
		# print(list(values))
		# print(f'selected {selected_lr}')
		return selected_lr

	# source: http://www.onmyphd.com/?p=gradient.descent
	@staticmethod
	def adapt_learning_rate_by_barzilai(delta_w, delta_gradients):
		selected_lambda = abs((delta_gradients.dot(delta_w.T)) / delta_gradients.dot(delta_gradients.T))
		return selected_lambda

	def __fit_with_gradient_descent(self, x, y, epochs, log_epoch=1000, gradients_to_stop=0, mse_to_stop=0,
									adapt_learning_rate=None, learning_rate_factor=1):
		x = LinearRegression.get_np_array(x)
		y = LinearRegression.get_np_array(y)

		self.x_mean = np.mean(x, axis=0)
		self.x_std = np.std(x, axis=0)
		x = self.normalize(x)
		x = LinearRegression.add_intercept(x)

		n, m = x.shape
		self.w = np.random.random((1, m))
		gradients = np.zeros(m)
		for i in range(epochs):
			prior_gradients = gradients
			prior_w = self.w

			predicted = x.dot(self.w.T)
			gradient_regularizator, error_regularizator = LinearRegression.get_ridge_gradient_and_error_regulazator(
				self.regularization_param, self.w, n)
			errors = predicted - y + gradient_regularizator
			gradients = errors.T.dot(x) / n

			self.w = self.w - self.learning_rate * gradients

			# adaptive lr
			if adapt_learning_rate == 'cauchy':
				self.learning_rate = self.adapt_learning_rate_by_cauchy(x, y, self.w, gradients)
			# elif adapt_learning_rate == 'barzilai':
			# 	self.learning_rate = self.adapt_learning_rate_by_barzilai(self.w - prior_w, gradients - prior_gradients)
			# elif adapt_learning_rate == 'factor':
			# 	self.learning_rate = self.adapt_learning_rate_by_factor(x, y, self.w, prior_w,
			# 															self.learning_rate, learning_rate_factor)


			mean_squared_error = LinearRegression.calculate_mean_squared_error(errors, error_regularizator)
			gradients_norm = abs(gradients).sum()

			if i % log_epoch == 0:
				print(f"Iter: {i}, gradients norm: {gradients_norm}, mse: {mean_squared_error}")
			if gradients_norm < gradients_to_stop or float(mean_squared_error) < mse_to_stop:
				break

	def __fit_with_stochastic_gd(self, x, y, epochs, log_epoch=1000, gradients_to_stop=0, mse_to_stop=0,
								 adapt_learning_rate=None, learning_rate_factor=1):
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
				if gradients_norm < gradients_to_stop or mean_square_error < mse_to_stop:
					break

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

	# model_lr = LinearRegression(0.01, regularization_param=1e-3, gradient_type='SGD')
	# model_lr.fit(x, y, epochs=int(1e3), log_epoch=100, gradients_to_stop=1e-6)

	model_lr = LinearRegression(0.1, regularization_param=1e-2, gradient_type='batch')
	model_lr.fit(x, y, epochs=int(1e4), log_epoch=100, gradients_to_stop=1e-6, adapt_learning_rate='cauchy')
	prediction = model_lr.predict(x)
	data['predicted'] = prediction
	print(data)
	print(f"MSE: {((data.iloc[:, -2] - data.iloc[:, -1])**2).sum() / data.shape[0]}")

	# data_new = pd.read_csv('../data/house_new.csv')
	# data_new['predicted'] = model_lr.predict(data_new)
	# print(data_new)


