import argparse
import math
import random
import sys
import os
import json
import numpy as np
import time

import torch

import arguments
import models
import models.data_utils.data_utils as data_utils
import models.model_utils as model_utils
from models.model import CodeGenerator

def create_model(args):
	model = CodeGenerator(args)
	if model.cuda_flag:
		model = model.cuda()
	model_supervisor = model_utils.Supervisor(model, args)
	if args.load_model:
		model_supervisor.load_pretrained(args.load_model)
	else:
		print('Created model with fresh parameters.')
		model_supervisor.model.init_weights(args.param_init)
	return model_supervisor

def train(args):
	print('Training:')

	data_processor = data_utils.DataProcessor(args)
	train_data = data_processor.load_data(0, args.train_size)
	val_data = data_processor.load_data(args.train_size, args.train_size + args.val_size)

	model_supervisor = create_model(args)

	for epoch in range(args.num_epochs):
		random.shuffle(train_data)
		for batch_idx in range(0, args.train_size, args.batch_size):
			print(epoch, batch_idx)
			batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties = data_processor.get_batch(train_data, args.batch_size, batch_idx)
			train_loss, train_acc = model_supervisor.train(batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties)
			print('train loss: %.4f train acc: %.4f' % (train_loss, train_acc))

			if model_supervisor.global_step % args.log_interval == 0:
				model_supervisor.save_model()

			if args.lr_decay_steps is not None and model_supervisor.global_step % args.lr_decay_steps == 0:
				model_supervisor.model.lr_decay(args.lr_decay_rate)

def evaluate(args):
	print('Evaluation:')
	data_processor = data_utils.DataProcessor(args)
	if args.iterative_retraining_prog_gen:
		test_data = data_processor.load_data(0, args.train_size)
	else:
		test_data = data_processor.load_data(args.train_size + args.val_size, args.train_size + args.val_size + args.test_size)
	model_supervisor = create_model(args)
	predictions = model_supervisor.eval(test_data)
	json.dump(predictions, open(args.pred_file + '.json', 'w'))

if __name__ == "__main__":
	arg_parser = arguments.get_arg_parser('restricted-c')
	args = arg_parser.parse_args()
	args.cuda = not args.cpu and torch.cuda.is_available()
	random.seed(args.seed)
	np.random.seed(args.seed)
	if args.iterative_retraining_prog_gen:
		args.eval = True
	if args.eval:
		evaluate(args)
	else:
		train(args)