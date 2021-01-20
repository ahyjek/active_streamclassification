#!"D:\Python38-32\python.exe"
import numpy as np
import random

class ActiveClassifierRandomStrategy:
	def __init__(self, classifier):
		self.classifier = classifier
		self.first_iteration = True
		
	def partial_fit(self, X, Y, classes=None):
		size = int(X.shape[0] * 0.02)
		selectedX = np.ndarray(shape=(size, X.shape[1]))
		selectedY = np.ndarray(shape=(size,))
		used = []
		for i in range(size):
			j = random.randint(0, size)
			while j in used:
				j = random.randint(0, size)
			used.append(j)
			selectedX[i] = X[j]
			selectedY[i] = Y[j]
		self.classifier.partial_fit(selectedX, selectedY, classes)
		
	def predict(self, X):
		return self.classifier.predict(X)


class ActiveClassifierVariableUncertaintyStrategy:
	def __init__(self, classifier):
		self.classifier = classifier
		self.first_iteration = True
		
	def partial_fit(self, X, Y, classes=None):
		for rp in range(5):
			size = int(X.shape[0] * 0.004)
			if self.first_iteration:
				self.first_iteration = False
				selectedX = np.ndarray(shape=(size, X.shape[1]))
				selectedY = np.ndarray(shape=(size,))
				used = []
				for i in range(size):
					j = random.randint(0, size)
					while j in used:
						j = random.randint(0, size)
					used.append(j)
					selectedX[i] = X[j]
					selectedY[i] = Y[j]
				self.classifier.partial_fit(selectedX, selectedY, classes)
			else:
				y_prob = self.classifier.predict_proba(X)
				var = np.var(y_prob, 1)
				selectedX = np.ndarray(shape=(size, X.shape[1]))
				selectedY = np.ndarray(shape=(size,))
			
				ids = [] 
				for id in range(X.shape[0]):
					ids.append(id)
				ids = sorted(ids, key= lambda id: var[id])
				for i in range(size):
					selectedX[i] = X[ids[i]]
					selectedY[i] = Y[ids[i]]
				self.classifier.partial_fit(selectedX, selectedY, classes)
		
	def predict(self, X):
		return self.classifier.predict(X)

class ActiveClassifierVariableUncertaintyStrategyWithRandomization:
	def __init__(self, classifier):
		self.classifier = classifier
		self.first_iteration = True
		
	def partial_fit(self, X, Y, classes=None):
		for rp in range(5):
			size = int(X.shape[0] * 0.004)
			if self.first_iteration:
				self.first_iteration = False
				selectedX = np.ndarray(shape=(size, X.shape[1]))
				selectedY = np.ndarray(shape=(size,))
				used = []
				for i in range(size):
					j = random.randint(0, size)
					while j in used:
						j = random.randint(0, size)
					used.append(j)
					selectedX[i] = X[j]
					selectedY[i] = Y[j]
				self.classifier.partial_fit(selectedX, selectedY, classes)
			else:
				y_prob = self.classifier.predict_proba(X)
				var = np.var(y_prob, 1)
				selectedX = np.ndarray(shape=(size, X.shape[1]))
				selectedY = np.ndarray(shape=(size,))
			
				ids = [] 
				for id in range(X.shape[0]):
					ids.append(id)
				ids = sorted(ids, key= lambda id: var[id])
				values = []
				for i in range(size):
					value = None
					while value is None or value in values:
						value = int(abs(random.normalvariate(0,X.shape[0]/3)))
						if value >= X.shape[0]:
							value = None
					values.append(value)
					selectedX[i] = X[ids[value]]
					selectedY[i] = Y[ids[value]]
				self.classifier.partial_fit(selectedX, selectedY, classes)
		
	def predict(self, X):
		return self.classifier.predict(X)
