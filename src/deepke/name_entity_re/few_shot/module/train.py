import torch
from torch import optim
from tqdm import tqdm
from ..utils import convert_preds_to_outputs, write_predictions
import random

class Trainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, process=None, args=None, logger=None, loss=None, metrics=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.process = process
        self.logger = logger
        self.metrics = metrics
        self.writer = writer
        self.loss = loss
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.eval_begin_epoch = args.eval_begin_epoch
        self.device = args.device
        self.load_path = args.load_path
        self.save_path = args.save_path
        self.refresh_step = 1
        self.best_metric = 0
        self.best_dev_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * self.num_epochs
        self.step = 0
        self.args = args
        
    
    def train(self):
        self.before_train()  # something should do before training
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.batch_size)
        self.logger.info("  Num epoch = %d", self.num_epochs)
        self.logger.info("  Batch size = %d", self.batch_size)
        self.logger.info("  Learning rate = {}".format(self.lr))
        self.logger.info("  Evaluate begin = %d", self.eval_begin_epoch)

        if self.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.load_path))
            load_model_dict = torch.load(self.args.load_path)
            model_dict = self.model.state_dict()
            for name in load_model_dict:
                if name in model_dict:
                    if model_dict[name].shape == load_model_dict[name].shape:
                        model_dict[name] = load_model_dict[name]
                    else:
                        self.logger.info(f"Skip loading parameter: {name}, "
                            f"required shape: {model_dict[name].shape}, "
                            f"loaded shape: {load_model_dict[name].shape}")
                else:
                    self.logger.info(f"Not Found! Skip loading parameter: {name}.")
            self.model.load_state_dict(model_dict)
            self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(self.num_epochs):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    loss = self._step(batch, mode="train")
                    avg_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(1)
                        pbar.set_postfix_str(print_output)            
                        # self.writer.add_scalar(tag='loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        self.writer.log({ 'avg_loss':avg_loss})
                        avg_loss = 0
                        
                if epoch >= self.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.
            pbar.close()
            self.pbar = None
            self.logger.info("Get best performance at epoch {}, best f1 score is {:.2f}".format(self.best_dev_epoch, self.best_metric))



    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.batch_size)
        self.logger.info("  Batch size = %d", self.batch_size)
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                for batch in self.dev_data:
                    batch = (tup.to(self.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    self._step(batch, mode="dev")
                    pbar.update()
                # evaluate done
                eva_result = self.metrics.get_metric()
                pbar.close()
                self.logger.info("Epoch {}/{}, best f1: {}, current f1 score: {:.2f}, recall: {:.2f}, precision: {:.2f}."\
                            .format(epoch, self.num_epochs, self.best_metric, eva_result['f'], eva_result['rec'], eva_result['pre']))
                # self.writer.add_scalars('evaluate', {'f1': eva_result['f'],
                #                                     'recall': eva_result['rec'],
                #                                     'precision': eva_result['pre']}, epoch)
                self.writer.log({'f1': eva_result['f'],'recall': eva_result['rec'],'precision': eva_result['pre']})

                if eva_result['f'] >= self.best_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_metric = eva_result['f']  # update best metric(f1 score)
                    if self.save_path is not None:      # need to save model
                        torch.save(self.model.state_dict(), self.save_path+"/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.save_path))

        self.model.train()

    def predict(self):
        self.model.eval()
        self.logger.info("***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.batch_size)
        self.logger.info("  Batch size = %d", self.batch_size)
        if self.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.load_path))
            self.model.load_state_dict(torch.load(self.load_path))
            self.logger.info("Load model successful!")
            self.model.to(self.device)

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Test")
                texts = []
                labels = []
                for batch in self.test_data:
                    batch = (tup.to(self.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    src_tokens, src_seq_len, first, raw_words = batch
                    preds = self._step((src_tokens, src_seq_len, first), mode="test")
                    outputs = convert_preds_to_outputs(preds, raw_words, self.process.mapping, self.process.tokenizer)
                    texts.extend(raw_words)
                    labels.extend(outputs)
                    pbar.update()

        self.logger.info("***** Predict example *****")
        idx = random.randint(0, len(texts))
        print(len(texts), len(labels))
        self.logger.info("Raw texts: " + " ".join(texts[idx]))
        self.logger.info("Prediction: " + " ".join(labels[idx]))
        if self.args.write_path is not None:    # write predict
            write_predictions(self.args.write_path, texts, labels)
            self.logger.info("Write into {}!".format(self.args.write_path))

        

    def _step(self, batch, mode="train"):
        if mode=="dev": # dev: compute metric
            src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, target_span = batch
            pred = self.model.predict(src_tokens, src_seq_len, first)
            self.metrics.evaluate(target_span, pred, tgt_tokens)
            return
        elif mode=="test":  # test: just get pred
            src_tokens, src_seq_len, first = batch
            pred = self.model.predict(src_tokens, src_seq_len, first)
            return pred
        else:   # train: get loss
            src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, target_span = batch
            pred = self.model(src_tokens, tgt_tokens, src_seq_len, first)
            loss = self.loss(tgt_tokens, tgt_seq_len, pred)
            return loss


    def before_train(self):
        parameters = []
        params = {'lr':self.lr, 'weight_decay':1e-2}
        params['params'] = [param for name, param in self.model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
        parameters.append(params)

        params = {'lr':self.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
                params['params'].append(param)
        parameters.append(params)

        params = {'lr':self.lr, 'weight_decay':0}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
                params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters)

        if self.args.freeze_plm:    # freeze pretrained language model(bart)
            for name, par in self.model.named_parameters():
                if 'prompt_encoder' in name or 'prompt_decoder' in name and "bart_mlp" not in name:
                    par.requires_grad = False

        self.model.to(self.device)
