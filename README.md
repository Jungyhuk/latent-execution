# Latent Execution for Neural Program Synthesis

This repo provides the code to replicate the experiments in the paper

> Xinyun Chen, Dawn Song, Yuandong Tian, <cite> Latent Execution for Neural Program Synthesis, in NeurIPS 2021. </cite>

Paper [[arXiv](https://arxiv.org/abs/2107.00101)] [[NeurIPS](https://openreview.net/pdf?id=_nRSyha2SP)]

## Prerequisites

[PyTorch](https://pytorch.org)

[Dataset](https://drive.google.com/file/d/11W4VTwBDN0Un_98IJr3231Ev1t9kkp7A/view?usp=sharing)

## Sample Usage

1. To run our full latent program synthesizer (LaSynth):

``python run.py --latent_execution --operation_predictor --decoder_self_attention``

2. To run our program synthesizer without partial program execution (NoPartialExecutor):

``python run.py --latent_execution --operation_predictor --decoder_self_attention --no_partial_execution``

3. To run the RobustFill model:

``python run.py``

4. To run the Property Signatures model:

``python run.py --use_properties``

## Run experiments

In the following we list some important arguments for experiments:
* `--data_folder`: path to the dataset.
* `--model_dir`: path to the directory that stores the models.
* `--load_model`: path to the pretrained model (optional).
* `--eval`: adding this command will enable the evaluation mode; otherwise, the model will be trained by default.
* `--num_epochs`: number of training epochs. The default value is `10`, but usually 1 epoch is enough for a decent performance.
* `--log_interval LOG_INTERVAL`: saving checkpoints every `LOG_INTERVAL` steps.
* `--latent_execution`: Enable the model to learn the latent executor module.
* `--no_partial_execution`: Enable the model to learn the latent executor module, but this module is not used by the program synthesizer, and only adds to the training loss.
* `--operation_predictor`: Enable the model to learn the operation predictor module.
* `--use_properties`: Run the Property Signatures baseline.
* `--iterative_retraining_prog_gen`: Decode training programs for iterative retraining.

More details can be found in ``arguments.py``.

## Citation

If you use the code in this repo, please cite the following paper:

```
@inproceedings{chen2021latent,
  title={Latent Execution for Neural Program Synthesis},
  author={Chen, Xinyun and Song, Dawn and Tian, Yuandong},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
## License
This repo is CC-BY-NC licensed, as found in the [LICENSE file](./LICENSE).

## References

[1] Devlin et al., RobustFill: Neural Program Learning under Noisy I/O, ICML 2017.

[2] Odena and Sutton, Learning to Represent Programs with Property Signatures, ICLR 2020.

[3] Chen et al., Execution-Guided Neural Program Synthesis, ICLR 2019.