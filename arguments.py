import argparse
import time
import os
import sys

def get_arg_parser(title):
	parser = argparse.ArgumentParser(description=title)
	parser.add_argument('--cpu', action='store_true', default=False)
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--model_dir', type=str, default='../checkpoints/model_0')
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--num_LSTM_layers', type=int, default=2)
	parser.add_argument('--num_MLP_layers', type=int, default=1)
	parser.add_argument('--num_attention_layers', type=int, default=2)
	parser.add_argument('--LSTM_hidden_size', type=int, default=512)
	parser.add_argument('--MLP_hidden_size', type=int, default=512)
	parser.add_argument('--embedding_size', type=int, default=1024)
	parser.add_argument('--tokenizer_name', type=str, default='microsoft/codebert-base')
	parser.add_argument('--latent_execution', action='store_true')
	parser.add_argument('--operation_predictor', action='store_true')
	parser.add_argument('--decoder_self_attention', action='store_true')
	parser.add_argument('--use_properties', action='store_true')
	parser.add_argument('--no_partial_execution', action='store_true')

	parser.add_argument('--keep_last_n', type=int, default=None)
	parser.add_argument('--log_interval', type=int, default=5000)
	parser.add_argument('--pred_file', type=str, default='predictions')

	parser.add_argument('--iterative_retraining_prog_gen', action='store_true')
	parser.add_argument('--iterative_retraining_prog_gen_folder', type=str, default='../data_v2')
	parser.add_argument('--iterative_retraining_prog_gen_file', type=str, default='code')

	data_group = parser.add_argument_group('data')
	data_group.add_argument('--max_input_len', type=int, default=50)
	data_group.add_argument('--max_decode_len', type=int, default=256)
	data_group.add_argument('--exec_period', type=int, default=16)
	data_group.add_argument('--train_size', type=int, default=500000)
	data_group.add_argument('--val_size', type=int, default=1000)
	data_group.add_argument('--test_size', type=int, default=1000)
	data_group.add_argument('--data_folder', type=str, default='../data')
	data_group.add_argument('--data_filename', type=str, default='code_')
	data_group.add_argument('--io_size', type=int, default=5)
	data_group.add_argument('--num_vars', type=int, default=5)
	data_group.add_argument('--value_range', type=int, default=4)

	train_group = parser.add_argument_group('train')
	train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
	train_group.add_argument('--lr', type=float, default=1e-3)
	train_group.add_argument('--lr_decay_steps', type=int, default=6000)
	train_group.add_argument('--lr_decay_rate', type=float, default=0.9)
	train_group.add_argument('--dropout_rate', type=float, default=0.0)
	train_group.add_argument('--gradient_clip', type=float, default=5.0)
	train_group.add_argument('--num_epochs', type=int, default=50)
	train_group.add_argument('--batch_size', type=int, default=8)
	train_group.add_argument('--param_init', type=float, default=0.1)
	train_group.add_argument('--seed', type=int, default=None)

	return parser