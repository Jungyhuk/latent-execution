import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MLPModel(nn.Module):
	"""
	Multi-layer perception module.
	"""
	def __init__(self, num_layers, input_size, hidden_size, output_size, dropout_rate, cuda_flag, activation=None):
		super(MLPModel, self).__init__()
		self.num_layers = num_layers
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.cuda_flag = cuda_flag
		self.dropout_rate = dropout_rate
		self.model = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.Dropout(p=self.dropout_rate),
			nn.ReLU())
		for _ in range(self.num_layers):
			self.model = nn.Sequential(
				self.model,
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.Dropout(p=self.dropout_rate),
				nn.ReLU())
		self.model = nn.Sequential(
			self.model,
			nn.Linear(self.hidden_size, self.output_size))
		if activation is not None:
			self.model = nn.Sequential(
				self.model,
				activation
				)

	def forward(self, inputs):
		return self.model(inputs)