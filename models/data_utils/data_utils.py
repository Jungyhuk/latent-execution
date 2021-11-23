import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip

import torch
from torch.autograd import Variable
import transformers

_PAD = b"_PAD"
_EOS = b"_EOS"
_GO = b"_GO"
_START_VOCAB = [_PAD, _EOS, _GO]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2

def np_to_tensor(inp, output_type, cuda_flag, eval_flag=False):
	if eval_flag:
		with torch.no_grad():
			if output_type == 'float':
				inp_tensor = Variable(torch.FloatTensor(inp))
			elif output_type == 'int':
				inp_tensor = Variable(torch.LongTensor(inp))
			else:
				print('undefined tensor type')
	else:
		if output_type == 'float':
			inp_tensor = Variable(torch.FloatTensor(inp))
		elif output_type == 'int':
			inp_tensor = Variable(torch.LongTensor(inp))
		else:
			print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor.cuda()
	return inp_tensor

class DataProcessor(object):
	def __init__(self, args):
		self.tokenizer_name = args.tokenizer_name
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
		self.train_size = args.train_size
		self.val_size = args.val_size
		self.test_size = args.test_size
		self.io_size = args.io_size
		self.num_vars = args.num_vars
		self.data_folder = args.data_folder
		self.data_filename = args.data_filename
		self.max_input_len = args.max_input_len
		self.max_decode_len = args.max_decode_len
		self.cuda_flag = args.cuda
		self.eval_flag = args.eval
		self.value_range = args.value_range
		self.value_offset = self.value_range * 4

	def load_data(self, st_index, ed_index):
		data = []
		for sample_index in range(st_index, ed_index):
			file_name = os.path.join(self.data_folder, self.data_filename + str(sample_index))
			c_file = file_name + '.c'
			fin = open(c_file, 'r')
			c_code = fin.read()
			fin.close()
			init_var_list = []
			a_list = ['a' + str(i) for i in range(self.num_vars)]
			for var_index, var_name in enumerate(a_list):
				var_assignment_str = var_name + '[' + str(self.io_size) + '] = {'
				var_assignment_index = c_code.index(var_assignment_str)
				var_assignment_index += len(var_assignment_str)
				while c_code[var_assignment_index] in [' ', '\n']:
					var_assignment_index += 1
				var_assignment = c_code[var_assignment_index:]
				var_assignment = var_assignment[:var_assignment.index('};\n')]
				var_assignment = var_assignment.split(', ')
				for i in range(len(var_assignment)):
					if var_assignment[i][0] == '(' and var_assignment[i][-1] == ')':
						var_assignment[i] = var_assignment[i][1:-1]
				init_var_list.append(var_assignment)
			func_head = 'int * func_1(int a[])'
			gt_code = c_code[c_code.index(func_head) + len(func_head):]
			gt_code = gt_code[gt_code.index(func_head) + len(func_head):]
			while gt_code[0] != '{':
				gt_code = gt_code[1:]
			gt_code = gt_code[:gt_code.index('int main(void)')]
			gt_code = gt_code.strip()
			gt_code = gt_code.split('\n')
			for line_idx in range(len(gt_code)):
				gt_code[line_idx] = gt_code[line_idx].strip()
			gt_code = '\n'.join(gt_code)
			out_file = file_name + '.out'
			fin = open(out_file, 'r')
			out = fin.read()
			out = out.split('\n')
			out = out[:-1]
			fin.close()
			inp_var_list = []
			outp_var_list = []
			cur_data = {}
			out_list = []
			for io_index in range(len(init_var_list[0])):
				out_list.append(out[io_index].split(' '))

			properties = []
			for var_index in range(len(init_var_list)):
				for delta in range(-self.value_range, self.value_range + 1):
					flag = (int(out_list[0][var_index]) == int(init_var_list[0][var_index]) + delta)
					for io_index in range(1, len(init_var_list[0])):
						cur_flag = (int(out_list[io_index][var_index]) == int(init_var_list[io_index][var_index]) + delta)
						if cur_flag != flag:
							flag = 2
							break
					flag = int(flag)
					properties.append(flag)
			for io_index in range(len(init_var_list[0])):
				cur_inp_var_list = []
				for var_index in range(len(init_var_list)):
					cur_inp_var_list += [int(init_var_list[io_index][var_index]) + self.value_offset]
					if var_index == len(init_var_list) - 1:
						break
				inp_var_list.append(cur_inp_var_list)
				cur_outp_var_list = out[io_index].split(' ')
				cur_outp_var_list = cur_outp_var_list[:-1]
				cur_outp_var_list = [int(tok) + self.value_offset for tok in cur_outp_var_list]
				outp_var_list.append(cur_outp_var_list)
			cur_data['inp_var_list'] = inp_var_list
			cur_data['outp_var_list'] = outp_var_list
			cur_data['init_gt_code'] = gt_code
			gt_code = self.tokenizer.encode(gt_code)
			while gt_code[-1] == self.tokenizer.eos_token_id:
				gt_code = gt_code[:-1]
			while gt_code[0] == 0:
				gt_code = gt_code[1:]
			gt_code = gt_code[:self.max_decode_len - 1]
			gt_code += [self.tokenizer.eos_token_id]
			cur_data['gt_code'] = gt_code
			cur_data['init_code'] = c_code
			cur_data['file_name'] = file_name
			cur_data['properties'] = properties
			data.append(cur_data)
		return data

	def get_batch(self, data, batch_size, start_idx):
		data_size = len(data)
		batch_inp_var_list = []
		batch_outp_var_list = []
		batch_gt_code = []
		max_var_list_len = 0
		max_code_len = 0
		batch_properties = []
		for idx in range(start_idx, min(start_idx + batch_size, data_size)):
			sample = data[idx]
			cur_batch_inp_var_list = []
			cur_batch_outp_var_list = []
			for io_idx in range(len(sample['inp_var_list'])):
				cur_batch_inp_var_list.append(sample['inp_var_list'][io_idx].copy())
				cur_batch_outp_var_list.append(sample['outp_var_list'][io_idx].copy())
				max_var_list_len = max(max_var_list_len, len(sample['inp_var_list'][io_idx]))
				max_var_list_len = max(max_var_list_len, len(sample['outp_var_list'][io_idx]))
			batch_inp_var_list.append(cur_batch_inp_var_list)
			batch_outp_var_list.append(cur_batch_outp_var_list)
			batch_gt_code.append(sample['gt_code'].copy())
			max_code_len = max(max_code_len, len(sample['gt_code']))
			batch_properties.append(sample['properties'].copy())
		for idx in range(len(batch_inp_var_list)):
			for io_idx in range(len(batch_inp_var_list[idx])):
				batch_inp_var_list[idx][io_idx] += [0] * (max_var_list_len - len(batch_inp_var_list[idx][io_idx]))
				batch_outp_var_list[idx][io_idx] += [0] * (max_var_list_len - len(batch_outp_var_list[idx][io_idx]))
			batch_gt_code[idx] += [0] * (max_code_len - len(batch_gt_code[idx]))
		batch_inp_var_list = np.array(batch_inp_var_list)
		batch_outp_var_list = np.array(batch_outp_var_list)
		batch_gt_code = np.array(batch_gt_code)
		batch_inp_var_list = np_to_tensor(batch_inp_var_list, 'int', self.cuda_flag, self.eval_flag)
		batch_outp_var_list = np_to_tensor(batch_outp_var_list, 'int', self.cuda_flag, self.eval_flag)
		batch_gt_code = np_to_tensor(batch_gt_code, 'int', self.cuda_flag, self.eval_flag)
		batch_properties = np.array(batch_properties)
		batch_properties = np_to_tensor(batch_properties, 'int', self.cuda_flag, self.eval_flag)
		return batch_inp_var_list, batch_outp_var_list, batch_gt_code, batch_properties
