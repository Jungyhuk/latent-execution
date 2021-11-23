import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
import transformers

import numpy as np
from .data_utils import data_utils
from .modules import mlp
import transformers

class CodeGenerator(nn.Module):
	def __init__(self, args):
		super(CodeGenerator, self).__init__()
		self.cuda_flag = args.cuda
		self.eval_flag = args.eval
		self.tokenizer_name = args.tokenizer_name
		self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
		self.vocab_size = len(self.tokenizer)
		self.batch_size = args.batch_size
		self.embedding_size = args.embedding_size
		self.LSTM_hidden_size = args.LSTM_hidden_size
		self.MLP_hidden_size = args.MLP_hidden_size
		self.num_LSTM_layers = args.num_LSTM_layers
		self.num_MLP_layers = args.num_MLP_layers
		self.num_attention_layers = args.num_attention_layers
		self.gradient_clip = args.gradient_clip
		self.lr = args.lr
		self.dropout_rate = args.dropout_rate
		self.max_input_len = args.max_input_len
		self.max_decode_len = args.max_decode_len
		self.io_size = args.io_size
		self.exec_period = args.exec_period
		self.dropout = nn.Dropout(p=self.dropout_rate)
		self.ceLoss = nn.CrossEntropyLoss()
		self.mseLoss = nn.MSELoss()
		self.latent_execution = args.latent_execution
		self.no_partial_execution = args.no_partial_execution
		self.operation_predictor = args.operation_predictor
		self.value_range = args.value_range
		self.value_offset = self.value_range * 4
		self.var_offset = -self.value_range + self.value_offset
		self.decoder_self_attention_flag = args.decoder_self_attention
		self.use_properties = args.use_properties

		self.code_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
		if self.decoder_self_attention_flag or self.use_properties:
			self.code_predictor = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, self.vocab_size, self.dropout_rate, self.cuda_flag)
		elif self.operation_predictor:
			self.code_predictor = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 6, self.MLP_hidden_size, self.vocab_size, self.dropout_rate, self.cuda_flag)
		else:
			self.code_predictor = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 4, self.MLP_hidden_size, self.vocab_size, self.dropout_rate, self.cuda_flag)
		self.var_embedding = nn.Embedding(self.value_offset * 2 + 1, self.embedding_size)

		self.input_var_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
		self.output_var_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
		self.operation_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)

		self.decoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
		self.prog_executor = nn.LSTM(input_size=self.LSTM_hidden_size * 2, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)

		self.prog_executor_attention = nn.Linear(self.LSTM_hidden_size * 2, self.embedding_size)

		self.encoder_o2i_attention_linear = nn.ModuleList([nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2) for _ in range(self.num_attention_layers)])
		self.encoder_i2o_attention_linear = nn.ModuleList([nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2) for _ in range(self.num_attention_layers)])
		self.decoder_d2i_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.decoder_d2o_attention_linear = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)
		self.decoder_i2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.decoder_o2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.decoder_d2op_attention_linear = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)
		self.decoder_op2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
		self.encoder_self_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)

		if self.operation_predictor:
			self.decoder_self_attention_linear = nn.Linear(self.LSTM_hidden_size * 6, self.LSTM_hidden_size * 2)
		else:
			self.decoder_self_attention_linear = nn.Linear(self.LSTM_hidden_size * 4, self.LSTM_hidden_size * 2)
		self.attention_tanh = nn.Tanh()

		self.keys = [int(x + self.value_offset) for x in range(-self.value_range, self.value_range + 1)]
		self.key_attention_linear = nn.Linear(self.embedding_size, self.embedding_size)

		self.addition_values = []

		for x in range(-self.value_range, self.value_range + 1):
			self.addition_values.append([])
			for y in range(-self.value_range, self.value_range + 1):
				self.addition_values[-1].append(int(x + y + self.value_offset))

		self.subtract_values = []

		for x in range(-self.value_range, self.value_range + 1):
			self.subtract_values.append([])
			for y in range(-self.value_range, self.value_range + 1):
				self.subtract_values[-1].append(int(y - x + self.value_offset))

		self.values = self.addition_values + self.subtract_values

		self.keys = np.array(self.keys)
		self.keys = data_utils.np_to_tensor(self.keys, 'int', self.cuda_flag, self.eval_flag)

		self.values = np.array(self.values)
		self.values = data_utils.np_to_tensor(self.values, 'int', self.cuda_flag, self.eval_flag)

		self.value_row_cnt = self.values.size()[0]
		self.value_col_cnt = self.values.size()[1]
		self.values = self.values.reshape(-1)
		self.operation_embedding = nn.Embedding(self.value_row_cnt, self.embedding_size)

		if self.use_properties:
			self.property_embedding = nn.Embedding(3, self.embedding_size)
			self.property_encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, dropout=self.dropout_rate, bidirectional=True, batch_first=True)
			self.decoder_p2d_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)
			self.decoder_d2p_attention_linear = nn.Linear(self.LSTM_hidden_size * 2, self.LSTM_hidden_size * 2)			

		if args.optimizer == 'adam':
			self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		elif args.optimizer == 'sgd':
			self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
		elif args.optimizer == 'rmsprop':
			self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
		else:
			raise ValueError('optimizer undefined: ', args.optimizer)

	def init_weights(self, param_init):
		for param in self.parameters():
			nn.init.uniform_(param, -param_init, param_init)

	def lr_decay(self, lr_decay_rate):
		self.lr *= lr_decay_rate
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def train_step(self):
		if self.gradient_clip > 0:
			clip_grad_norm(self.parameters(), self.gradient_clip)
		self.optimizer.step()

	def attention(self, encoder_outputs, decoder_output, encoder_attention_linear, decoder_attention_linear):
		decoder_output_linear = decoder_attention_linear(decoder_output)
		encoder_outputs_linear = encoder_attention_linear(encoder_outputs)
		if len(decoder_output_linear.size()) < 3:
			decoder_output_linear = decoder_output_linear.unsqueeze(1)
		dotted = torch.bmm(encoder_outputs_linear, torch.transpose(decoder_output_linear, 1, 2))
		decoder_output_linear = decoder_output_linear.squeeze(1)
		dotted = dotted.squeeze(2)
		attention = nn.Softmax(dim=1)(dotted)
		if len(attention.size()) < 3:
			attention = attention.unsqueeze(2)
		encoder_attention = torch.bmm(torch.transpose(encoder_outputs_linear, 1, 2), attention)
		encoder_attention = encoder_attention.squeeze(2)
		if len(encoder_attention.size()) == 3:
			encoder_attention = torch.transpose(encoder_attention, 1, 2)
		res = self.attention_tanh(encoder_attention + decoder_output_linear)
		return res


	def prog_exec(self, input_var_encoder_outputs, init_exec_hidden_state):
		var_list_len = input_var_encoder_outputs.size()[1]
		exec_var_encoder_outputs, _ = self.prog_executor(input_var_encoder_outputs, init_exec_hidden_state)
		return exec_var_encoder_outputs

	def forward(self, batch_input_var_list, batch_output_var_list, batch_gt_code, batch_properties, eval_flag=False):
		batch_size = batch_gt_code.size()[0]
		input_var_len = batch_input_var_list.size()[-1]
		output_var_len = batch_output_var_list.size()[-1]

		if self.operation_predictor:
			key_embedding = self.var_embedding(self.keys)
			key_embedding = self.key_attention_linear(key_embedding)

			value_embedding = self.var_embedding(self.values)
			value_embedding = self.key_attention_linear(value_embedding)

		if self.use_properties:
			property_embedding = self.property_embedding(batch_properties)
			property_encoder_outputs, property_encoder_hidden_state = self.property_encoder(property_embedding)

		else:
			batch_input_var_list = batch_input_var_list.reshape(batch_size * self.io_size, -1)
			batch_input_var_list_embedding = self.var_embedding(batch_input_var_list)

			batch_output_var_list = batch_output_var_list.reshape(batch_size * self.io_size, -1)
			batch_output_var_list_embedding = self.var_embedding(batch_output_var_list)
			
			input_var_encoder_outputs, input_var_encoder_hidden_state = self.input_var_encoder(batch_input_var_list_embedding)
			init_output_var_encoder_outputs, output_var_encoder_hidden_state = self.output_var_encoder(batch_output_var_list_embedding, input_var_encoder_hidden_state)

			output_var_encoder_outputs = init_output_var_encoder_outputs.clone()
			for i in range(self.num_attention_layers):
				output_var_encoder_outputs = self.attention(input_var_encoder_outputs, output_var_encoder_outputs, self.encoder_i2o_attention_linear[i], self.encoder_o2i_attention_linear[i])

		if self.operation_predictor:
			init_operation_row_attention = torch.matmul(batch_input_var_list_embedding, key_embedding.transpose(0, 1))
			init_operation_row_attention = init_operation_row_attention.reshape(batch_size * self.io_size * input_var_len, self.value_col_cnt, 1)
			init_operation_row_attention_weight = nn.Softmax(dim=-2)(init_operation_row_attention)
			gt_row_attention = batch_input_var_list.reshape(-1) - self.var_offset

			init_operation_col_attention = torch.matmul(batch_output_var_list_embedding, value_embedding.transpose(0, 1))
			init_operation_col_attention = init_operation_col_attention.reshape(batch_size * self.io_size * output_var_len, self.value_row_cnt, self.value_col_cnt)
			operation_col_attention = torch.bmm(init_operation_col_attention, init_operation_row_attention_weight)
			operation_col_attention = operation_col_attention.squeeze(-1)
			operation_col_attention_weight = nn.Softmax(dim=-1)(operation_col_attention)

			operations = data_utils.np_to_tensor(range(self.value_row_cnt), 'int', self.cuda_flag, eval_flag)
			operation_embedding = self.operation_embedding(operations)
			operation_embedding_per_token = torch.matmul(operation_col_attention_weight, operation_embedding)
			operation_embedding_per_token = operation_embedding_per_token.reshape(batch_size * self.io_size, output_var_len, self.embedding_size)

			operation_encoder_outputs, operation_encoder_hidden_state = self.input_var_encoder(operation_embedding_per_token)


		if self.use_properties:
			decoder_hidden_state = property_encoder_hidden_state
			decoder_input = torch.ones(batch_size, 1, dtype=torch.int64) * self.tokenizer.cls_token_id
		else: 
			decoder_hidden_state = (input_var_encoder_hidden_state[0] + output_var_encoder_hidden_state[0], input_var_encoder_hidden_state[1] + output_var_encoder_hidden_state[1])
			decoder_input = torch.ones(batch_size * self.io_size, 1, dtype=torch.int64) * self.tokenizer.cls_token_id

		if self.cuda_flag:
			decoder_input = decoder_input.cuda()
		decoder_input_embedding = self.code_embedding(decoder_input)
		logits = []
		predictions = []
		decode_history = None

		finished = torch.zeros(batch_size, 1, dtype=torch.int64)
		pad_mask = torch.zeros(self.vocab_size)
		pad_mask[data_utils.PAD_ID] = 1e9
		pad_mask = torch.stack([pad_mask] * batch_size, dim=0)
		if self.cuda_flag:
			finished = finished.cuda()
			pad_mask = pad_mask.cuda()

		if not eval_flag:
			decode_length = batch_gt_code.size()[1]
		else:
			decode_length = self.max_decode_len
		final_prog_exec_attention = None

		for step in range(decode_length):
			if decoder_hidden_state is not None:
				decoder_output, decoder_hidden_state = self.decoder(decoder_input_embedding, decoder_hidden_state)
			else:
				decoder_output, decoder_hidden_state = self.decoder(decoder_input_embedding)
			decoder_output = decoder_output.squeeze(1)
			if self.use_properties:
				decoder_output = self.attention(property_encoder_outputs, decoder_output, self.decoder_p2d_attention_linear, self.decoder_d2p_attention_linear)
			else:
				decoder_output_i = self.attention(input_var_encoder_outputs, decoder_output, self.decoder_i2d_attention_linear, self.decoder_d2i_attention_linear)
				decoder_output_o = self.attention(output_var_encoder_outputs, torch.cat([decoder_output, decoder_output_i], dim=-1), self.decoder_o2d_attention_linear, self.decoder_d2o_attention_linear)

				if self.operation_predictor:
					operation_attention = self.attention(operation_encoder_outputs, torch.cat([decoder_output_i, decoder_output_o], dim=-1), self.decoder_op2d_attention_linear, self.decoder_d2op_attention_linear)
					decoder_output = torch.cat([decoder_output_i, decoder_output_o, operation_attention], dim=-1)
				else:
					decoder_output = torch.cat([decoder_output_i, decoder_output_o], dim=-1)

				decoder_output = decoder_output.reshape(batch_size, self.io_size, -1)
				decoder_output, _ = decoder_output.max(1)

			if self.decoder_self_attention_flag:
				if decode_history is not None:
					decoder_output = self.attention(decode_history, decoder_output, self.encoder_self_attention_linear, self.decoder_self_attention_linear)
				else:
					decoder_output = self.decoder_self_attention_linear(decoder_output)
			cur_logits = self.code_predictor(decoder_output) + finished.float() * pad_mask
			logits.append(cur_logits)
			cur_predictions = cur_logits.max(1)[1]
			if eval_flag:
				decoder_input = cur_predictions
			else:
				decoder_input = batch_gt_code[:, step]
			predictions.append(decoder_input)
			values, indices = torch.sort(cur_logits, dim=1, descending=True)
			cur_finished = (decoder_input == self.tokenizer.eos_token_id).long().unsqueeze(1)
			finished = torch.max(finished, cur_finished)
			if torch.sum(finished) == batch_size:
				break
			decoder_input_embedding = self.code_embedding(decoder_input)
			decoder_input_embedding = decoder_input_embedding.unsqueeze(1)
			if decode_history is None:
				decode_history = decoder_input_embedding
			else:
				decode_history = torch.cat([decode_history, decoder_input_embedding], dim=1)
			if not self.use_properties:
				decoder_input = decoder_input.repeat_interleave(self.io_size)
			decoder_input_embedding = self.code_embedding(decoder_input)
			decoder_input_embedding = decoder_input_embedding.unsqueeze(1)
			if self.latent_execution and (((not self.no_partial_execution) and (step + 1) % self.exec_period == 0) or torch.sum((decoder_input == self.tokenizer.eos_token_id).int()) > 0):
				prog_exec_output = self.prog_exec(batch_input_var_list_embedding, decoder_hidden_state)
				prog_exec_attention = self.prog_executor_attention(prog_exec_output)
				variables = data_utils.np_to_tensor([int(x + self.value_offset) for x in range(-self.value_range, self.value_range + 1)], 'int', self.cuda_flag, eval_flag)
				var_embedding = self.var_embedding(variables)
				prog_exec_attention = torch.matmul(prog_exec_attention, var_embedding.transpose(0, 1))
				if final_prog_exec_attention is None:
					final_prog_exec_attention = prog_exec_attention
				else:
					eos_mask = (decoder_input == self.tokenizer.eos_token_id).float().unsqueeze(-1).unsqueeze(-1)
					final_prog_exec_attention = (1 - eos_mask) * final_prog_exec_attention + eos_mask * prog_exec_attention
				prog_exec_attention_weight = nn.Softmax(dim=-1)(prog_exec_attention)
				batch_input_var_list_embedding = torch.matmul(prog_exec_attention_weight, var_embedding)

				input_var_encoder_outputs, input_var_encoder_hidden_state = self.input_var_encoder(batch_input_var_list_embedding)

				output_var_encoder_outputs = init_output_var_encoder_outputs.clone()
				for i in range(self.num_attention_layers):
					output_var_encoder_outputs = self.attention(input_var_encoder_outputs, output_var_encoder_outputs, self.encoder_i2o_attention_linear[i], self.encoder_o2i_attention_linear[i])

				if self.operation_predictor:
					operation_row_attention = torch.matmul(batch_input_var_list_embedding, key_embedding.transpose(0, 1))
					operation_row_attention = operation_row_attention.reshape(batch_size * self.io_size * input_var_len, self.value_col_cnt, 1)
					operation_row_attention_weight = nn.Softmax(dim=-2)(operation_row_attention)

					operation_col_attention = torch.bmm(init_operation_col_attention, operation_row_attention_weight)
					operation_col_attention = operation_col_attention.squeeze(-1)
					operation_col_attention_weight = nn.Softmax(dim=-1)(operation_col_attention)

					operations = data_utils.np_to_tensor(range(self.value_row_cnt), 'int', self.cuda_flag, eval_flag)
					operation_embedding = self.operation_embedding(operations)
					operation_embedding_per_token = torch.matmul(operation_col_attention_weight, operation_embedding)
					operation_embedding_per_token = operation_embedding_per_token.reshape(batch_size * self.io_size, output_var_len, self.embedding_size)

					operation_encoder_outputs, operation_encoder_hidden_state = self.input_var_encoder(operation_embedding_per_token)

		predictions = torch.stack(predictions, dim=0)
		predictions = predictions.permute(1, 0)
		logits = torch.stack(logits, dim=0)
		logits = logits.permute(1, 2, 0)

		if self.operation_predictor:
			init_operation_row_attention = init_operation_row_attention.squeeze(-1)
		if self.latent_execution:
			final_prog_exec_attention = final_prog_exec_attention.reshape(batch_size * self.io_size * input_var_len, -1)
			gt_prog_exec_attention = batch_output_var_list.reshape(-1) - self.var_offset
		if eval_flag:
			if self.operation_predictor:
				total_loss = self.ceLoss(init_operation_row_attention, gt_row_attention)
			else:
				total_loss = 0.0
			if self.latent_execution:
				total_loss += self.ceLoss(final_prog_exec_attention, gt_prog_exec_attention)
		else:
			total_loss = self.ceLoss(logits, batch_gt_code)
			if self.operation_predictor:
				total_loss += self.ceLoss(init_operation_row_attention, gt_row_attention)
			if self.latent_execution:
				total_loss += self.ceLoss(final_prog_exec_attention, gt_prog_exec_attention)
		return total_loss, logits, predictions
