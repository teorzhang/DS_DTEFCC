from utils.CONSTANT import *
from transformers import TrainerCallback,TrainingArguments,TrainerState,TrainerControl
import logging
logger = logging.getLogger(__name__)


class FineTuneCallBack(TrainerCallback):

    def __init__(self,model_args):
        backbone_name = model_args.backbone_model.split('/')[-1]
        if backbone_name in {'bart_large', 'bart-large', 'bart_large_cnn', 'bart-large-cnn'}:
            self.p_name = BART_LARGE_PARAMS
        elif backbone_name in {'bart_base', 'bart-base'}:
            self.p_name = BART_BASE_PARAMS
        else:
            self.p_name = []
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        self.freeze = True        
        model = kwargs['model']
        for p in self.p_name:
            dict(model.named_parameters())[p].requires_grad = False
        return super().on_train_begin(args, state, control, **kwargs)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        
        #train_loss = kwargs['logs'].get('loss',None)
        #if train_loss is not None and self.freeze:
        if self.freeze:
            if state.global_step > 2000: 
                logger.info("*"*SCREEN_WIDTH)
                logger.info("Make all parameters trainable")
                for p in kwargs['model'].named_parameters():
                    if not p[1].requires_grad:
                        logger.info(p[0])
                        p[1].requires_grad = True
                self.freeze = False
        return super().on_epoch_begin(args, state, control, **kwargs)

class ShowModelParamsCallBack(TrainerCallback):
    """
    This class is to show model parameters in at the beginning of the training
    including:
    - num_params
    - parameters_unfreezed
    - parameters_freezed
    """
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        num_params_unfreezed = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_param_freezed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        num_param = num_params_unfreezed + num_param_freezed
        logger.info(f"The number of model parameters is : {num_param/(1000**2)}M")
        logger.info(f"The number of model parameters freezed is : {num_param_freezed/(1000**2)}M")
        logger.info(f"The number of model parameters unfreezed is : {num_params_unfreezed/(1000**2)}M")
        logger.info("*"*SCREEN_WIDTH)
        logger.info("Unfreezed Model Params:")
        for k in [k[0] for k in model.named_parameters() if k[1].requires_grad]:
            logger.info(k)
        logger.info("*"*SCREEN_WIDTH)
        logger.info("freezed Model Params:")
        for k in [k[0] for k in model.named_parameters() if not k[1].requires_grad]:
            logger.info(k)
        return super().on_train_begin(args, state, control, **kwargs)
