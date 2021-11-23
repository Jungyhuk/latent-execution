import numpy as np
import argparse
import sys
import os
import torch
import re
import json
import time

from torch.nn.utils import clip_grad_norm

from ..data_utils import data_utils

CKPT_PATTERN = re.compile('^ckpt-(\d+)$')


class Supervisor(object):
	def __init__(self, model, args):
		self.data_processor = data_utils.DataProcessor(args)
		self.model = model
		self.keep_last_n = args.keep_last_n
		self.global_step = 0
		self.batch_size = args.batch_size
		self.model_dir = args.model_dir
		self.pred_file = args.pred_file
		self.io_size = args.io_size
		self.iterative_retraining_prog_gen = args.iterative_retraining_prog_gen
		self.iterative_retraining_prog_gen_folder = args.iterative_retraining_prog_gen_folder
		self.iterative_retraining_prog_gen_file = args.iterative_retraining_prog_gen_file
		self.iterative_retraining_prog_gen_file = os.path.join(self.iterative_retraining_prog_gen_folder, self.iterative_retraining_prog_gen_file)
		self.train_size = args.train_size
		self.val_size = args.val_size


	def load_pretrained(self, load_model):
		print("Read model parameters from %s." % load_model)
		checkpoint = torch.load(load_model)
		self.model.load_state_dict(checkpoint)


	def save_model(self):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		global_step_padded = format(self.global_step, '08d')
		ckpt_name = 'ckpt-' + global_step_padded
		path = os.path.join(self.model_dir, ckpt_name)
		ckpt = self.model.state_dict()
		torch.save(ckpt, path)

		if self.keep_last_n is not None:
			ckpts = []
			for file_name in os.listdir(self.model_dir):
				matched_name = CKPT_PATTERN.match(file_name)
				if matched_name is None or matched_name == ckpt_name:
					continue
				step = int(matched_name.group(1))
				ckpts.append((step, file_name))
			if len(ckpts) > self.keep_last_n:
				ckpts.sort()
				os.unlink(os.path.join(self.model_dir, ckpts[0][1]))


	def train(self, batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties):
		self.model.optimizer.zero_grad()
		cur_loss, pred_logits, predictions = self.model(batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties)
		pred_acc = torch.sum(predictions == batch_gt_code)
		pred_acc = pred_acc.item() * 1.0 / (batch_gt_code.size()[0] * batch_gt_code.size()[1])

		self.global_step += 1
		cur_loss.backward()
		self.model.train_step()
		return cur_loss.item(), pred_acc

	def execute_code(self, file_name):
		os.system('gcc -o ' + file_name + ' ' + file_name + '.c')
		cmd = './' + file_name + ' > ' + file_name + '.out'
		os.system(cmd)

	def eval(self, data, max_eval_size=None):
		self.model.eval()
		data_size = len(data)
		if max_eval_size is not None:
			data_size = min(data_size, max_eval_size)
		eval_data = data[:data_size]
		test_acc = [0] * (self.io_size + 1)
		loss = 0
		
		predictions = []
		for batch_idx in range(0, data_size, self.batch_size):
			print(batch_idx)
			batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties = self.data_processor.get_batch(eval_data, self.batch_size, batch_idx)
			with torch.no_grad():
				cur_loss, init_cur_pred_logits, init_cur_predictions = self.model(batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties, eval_flag=True)
			loss += cur_loss * batch_gt_code.size()[0]
			cur_predictions = init_cur_predictions.data.cpu().numpy().tolist()
			for i in range(len(cur_predictions)):
				prog = self.data_processor.tokenizer.decode(cur_predictions[i])
				if '</s>' in prog:
					prog = prog[:prog.index('</s>')]
				predictions.append(prog)
				init_gt_code = eval_data[batch_idx + i]['init_gt_code']
				gt_code = self.data_processor.tokenizer.encode(init_gt_code)
				init_full_code = eval_data[batch_idx + i]['init_code']
				func_head = 'int * func_1(int a[])\n'
				pred_full_code = init_full_code[:init_full_code.index(func_head)] + func_head + prog + '\n' + init_full_code[init_full_code.index('int main(void)'):]
				if self.iterative_retraining_prog_gen:
					file_name = self.iterative_retraining_prog_gen_file + '_' + str(batch_idx + i)
				else:
					file_name = self.pred_file + '_' + str(batch_idx + i + self.train_size + self.val_size)
				fout = open(file_name + '.c', 'w')
				fout.write(pred_full_code)
				fout.close()
				self.execute_code(file_name)
				gt_file_name = eval_data[batch_idx + i]['file_name']
				fin = open(gt_file_name + '.out', 'r')
				gt_out = fin.read()
				fin.close()
				gt_out = gt_out.split('\n')
				gt_out = gt_out[:-1]
				if os.path.exists(file_name + '.out'):
					fin = open(file_name + '.out', 'r')
					pred_out = fin.read()
					fin.close()
					pred_out = pred_out.split('\n')
					pred_out = pred_out[:-1]
					if len(pred_out) != self.io_size:
						if self.iterative_retraining_prog_gen:
							os.system('cp ' + gt_file_name + ' ' + self.iterative_retraining_prog_gen_folder)
							os.system('cp ' + gt_file_name + '.c ' + self.iterative_retraining_prog_gen_folder)
							os.system('cp ' + gt_file_name + '.out ' + self.iterative_retraining_prog_gen_folder)
						continue
					cur_cnt = 0
					for io_idx in range(self.io_size):
						if gt_out[io_idx] == pred_out[io_idx]:
							cur_cnt += 1
					test_acc[cur_cnt] += 1
					if self.iterative_retraining_prog_gen and cur_cnt != self.io_size:
						os.system('cp ' + gt_file_name + ' ' + self.iterative_retraining_prog_gen_folder)
						os.system('cp ' + gt_file_name + '.c ' + self.iterative_retraining_prog_gen_folder)
						os.system('cp ' + gt_file_name + '.out ' + self.iterative_retraining_prog_gen_folder)
				elif self.iterative_retraining_prog_gen:
					os.system('cp ' + gt_file_name + ' ' + self.iterative_retraining_prog_gen_folder)
					os.system('cp ' + gt_file_name + '.c ' + self.iterative_retraining_prog_gen_folder)
					os.system('cp ' + gt_file_name + '.out ' + self.iterative_retraining_prog_gen_folder)
			del init_cur_predictions
			del init_cur_pred_logits
			torch.cuda.empty_cache()
		self.model.train()
		loss /= data_size
		print(loss)
		print(test_acc)
		return predictions
