
import torch
import torch.nn as nn



def make_optimizer(cfg, model, center_criterion=None):
	if not isinstance(model, nn.Module):
		raise TypeError("to build an optimizer, the model must be a type of nn.Module, but got type: '{}'".format(type(model)))
	
	
	params = []

	for key, value in model.named_parameters():
		if not value.requires_grad:
			continue

		lr = cfg.SOLVER.BASE_LR
		weight_decay = cfg.SOLVER.WEIGHT_DECAY
		if "bias" in key:
			lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
			weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
		params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
	

	if cfg.SOLVER.OPTIMIZER == "SGD":
		optimizer = getattr(torch.optim, "SGD")(params, momentum=cfg.SOLVER.MOMENTUM)
	else:
		optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER)(params)

	optimizer_center = None

	if center_criterion is not None:
		optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

	return optimizer, optimizer_center


