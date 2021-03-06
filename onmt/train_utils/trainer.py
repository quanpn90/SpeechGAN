from __future__ import division

import datetime
import gc
import inspect
import math
import os
import re
import time
import torch

import sys
from apex import amp
import copy
from torch import _VF

import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.multidata_iterator import MultiDataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def get_lambda(lambda_lat_dis, step):
    """
    Compute discriminators' lambdas.
    """
    s = 10000

    lambda_ = lambda_lat_dis * float(min(step, s)) / s

    return lambda_


def generate_data_iterator(dataset, seed, num_workers=1, epoch=1., buffer_size=0):
    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset
        # print("[INFO] Generating multi dataset iterator")
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size)
    else:

        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size)

    return data_iterator


class BaseTrainer(object):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.loss_function = loss_function
        self.start_time = 0

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def load_encoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        pretrained_model = build_model(checkpoint['opt'], checkpoint['dicts'])
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained encoder weights ...")
        pretrained_model.encoder.language_embedding = None
        enc_language_embedding = self.model.encoder.language_embedding
        self.model.encoder.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        self.model.encoder.load_state_dict(encoder_state_dict)
        self.model.encoder.language_embedding = enc_language_embedding
        return

    def load_decoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        chkpoint_dict = checkpoint['dicts']

        pretrained_model = build_model(checkpoint['opt'], chkpoint_dict)
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained decoder weights ...")
        # first we have to remove the embeddings which probably have difference size ...
        pretrained_word_emb = pretrained_model.decoder.word_lut
        pretrained_model.decoder.word_lut = None
        pretrained_lang_emb = pretrained_model.decoder.language_embeddings
        pretrained_model.decoder.language_embeddings = None

        # actually we assume that two decoders have the same language embeddings...
        untrained_word_emb = self.model.decoder.word_lut
        self.model.decoder.word_lut = None
        untrained_lang_emb = self.model.decoder.language_embeddings
        self.model.decoder.language_embeddings = None

        decoder_state_dict = pretrained_model.decoder.state_dict()
        self.model.decoder.load_state_dict(decoder_state_dict)

        # now we load the embeddings ....
        n_copies = 0
        for token in self.dicts['tgt'].labelToIdx:

            untrained_id = self.dicts['tgt'].labelToIdx[token]

            if token in chkpoint_dict['tgt'].labelToIdx:
                pretrained_id = chkpoint_dict['tgt'].labelToIdx[token]
                untrained_word_emb.weight.data[untrained_id].copy_(pretrained_word_emb.weight.data[pretrained_id])

                self.model.generator[0].linear.bias.data[untrained_id].copy_(pretrained_model
                                                                             .generator[0].linear.bias.data[
                                                                                 pretrained_id])
                n_copies += 1

        print("Copied embedding for %d words" % n_copies)
        self.model.decoder.word_lut = untrained_word_emb

        # now we load the language embeddings ...
        if pretrained_lang_emb and untrained_lang_emb and 'langs' in chkpoint_dict:
            for lang in self.dicts['langs']:

                untrained_id = self.dicts['langs'][lang]
                if lang in chkpoint_dict['langs']:
                    pretrained_id = chkpoint_dict['langs'][lang]
                    untrained_lang_emb.weight.data[untrained_id].copy_(pretrained_lang_emb.weight.data[pretrained_id])

        self.model.decoder.language_embeddings = untrained_lang_emb

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - '
                                   'try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(
                grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model.train()
        self.model.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        try:
            targets = batch.get('target_output')
            tgt_mask = None
            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                 zero_encoder=opt.zero_encoder,
                                 mirror=opt.mirror_loss, streaming_state=streaming_state,
                                 nce=opt.nce)

            outputs['tgt_mask'] = tgt_mask

            loss_dict = self.loss_function(outputs, targets, model=self.model)
            loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            if opt.mirror_loss:
                rev_loss = loss_dict['rev_loss']
                mirror_loss = loss_dict['mirror_loss']
                full_loss = full_loss + rev_loss + mirror_loss

            # reconstruction loss
            if opt.reconstruct:
                rec_loss = loss_dict['rec_loss']
                rec_loss = rec_loss
                full_loss = full_loss + rec_loss

            optimizer = self.optim.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)

                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             # print(varname(obj))
                #             # we can rule out parameter cost later
                #             # if 'parameter' not in type(obj):
                #             # if len(obj.shape) == 3:
                #             # if not isinstance(obj, torch.nn.parameter.Parameter):
                #             #     tensor = obj
                #             #     numel = tensor.
                #             print(type(obj), obj.type(), obj.size())
                #     except:
                #         pass

                # print("Memory profiling complete.")
                # print(torch.cuda.memory_summary())
                # exit()

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model.zero_grad()
            self.optim.zero_grad()
            # self.optim.step()
            # self.optim.reset()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()


class SpeechFNTrainer(object):
    def __init__(self, model, lat_dis, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):

        self.train_data = train_data
        self.valid_data = valid_data

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.start_time = 0
        self.n_gpus = len(self.opt.gpus)

        self.loss_function_ae, self.loss_lat_dis = loss_function
        self.model_ae = model
        self.lat_dis = lat_dis

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function_ae = self.loss_function_ae.cuda()
            self.model_ae = self.model_ae.cuda()
            self.lat_dis = self.lat_dis.cuda()
            self.loss_lat_dis = self.loss_lat_dis.cuda()
        if setup_optimizer:

            self.optim_ae = onmt.Optim(opt)
            self.optim_ae.set_parameters(self.model_ae.parameters())

            lat_opt = copy.deepcopy(opt)
            lat_opt.beta1 = 0.5
            # lat_opt.learning_rate = 0.0002
            # lat_opt.update_method = 'regular'
            self.optim_lat_dis = onmt.Optim(lat_opt)
            self.optim_lat_dis.set_parameters(self.lat_dis.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                # print(234)
                self.model_ae, self.optim_ae.optimizer = amp.initialize(self.model_ae,
                                                                        self.optim_ae.optimizer,
                                                                        opt_level=opt_level,
                                                                        keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                        loss_scale="dynamic",
                                                                        verbosity=1 if self.opt.verbose else 0)

                self.lat_dis, self.optim_lat_dis.optimizer = amp.initialize(self.lat_dis,
                                                                            self.optim_lat_dis.optimizer,
                                                                            opt_level=opt_level,
                                                                            keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                            loss_scale="dynamic",
                                                                            verbosity=1 if self.opt.verbose else 0)

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        print("Tacotron_warmup")
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model_ae.train()
        self.model_ae.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        try:
            encoder_outputs, decoder_outputs = self.model_ae(batch)

            gate_padded = batch.get('gate_padded')

            if self.opt.n_frames_per_step > 1:
                slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
                gate_padded = gate_padded[:, slice]

            src_org = batch.get('source_org')
            src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
            target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
            loss = self.loss_function_ae(decoder_outputs, target)
            full_loss = loss

            optimizer = self.optim_ae.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model_ae.zero_grad()
            self.optim_ae.zero_grad()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()

    def lat_dis_backward(self, batch):

        # self.model_ae.eval()
        self.model_ae.train()
        self.lat_dis.train()

        with torch.no_grad():
            encoder_outputs = self.model_ae.encode(batch)
            encoded_rep = encoder_outputs['context'].detach()
        preds = self.lat_dis(encoded_rep)

        loss = self.loss_lat_dis(preds, batch.get('source_lang'), mask=encoder_outputs['src_mask'], adversarial=False)

        loss_data = loss.data.item()
        # a little trick to avoid gradient overflow with fp16
        full_loss = loss

        optimizer = self.optim_lat_dis.optimizer

        if self.cuda:
            with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            full_loss.backward()

        return loss_data, encoder_outputs

    def autoencoder_backward(self, batch, step=0):

        self.model_ae.train()
        self.lat_dis.eval()
        encoder_outputs, decoder_outputs = self.model_ae(batch)

        gate_padded = batch.get('gate_padded')

        if self.opt.n_frames_per_step > 1:
            slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
            gate_padded = gate_padded[:, slice]

        src_org = batch.get('source_org')
        src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
        target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
        loss = self.loss_function_ae(decoder_outputs, target)

        if True:
            lat_dis_preds = self.lat_dis(encoder_outputs['context'])
            adversarial_loss = self.loss_lat_dis(lat_dis_preds, batch.get('source_lang'),
                                                 mask=encoder_outputs['src_mask'], adversarial=True)

            lambda_ = get_lambda(self.opt.lambda_lat_dis_coeff, step)
            loss = loss + lambda_ * adversarial_loss  # lambda

        loss_data = loss.data.item()
        adversarial_loss_data = adversarial_loss.data.item()
        # a little trick to avoid gradient overflow with fp16
        full_loss = loss

        optimizer = self.optim_ae.optimizer

        if self.cuda:
            with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            full_loss.backward()

        return loss_data, adversarial_loss_data, encoder_outputs

    def save(self, epoch, valid_loss, itr=None):

        opt = self.opt
        model = self.model_ae
        dicts = self.dicts

        model_state_dict = (self.model_ae.state_dict(), self.lat_dis.state_dict())
        optim_state_dict = (self.optim_ae.state_dict(), self.optim_lat_dis.state_dict())

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        iterations = self.optim_ae._step
        file_name = '%s_loss_%.6f_i%.2f.pt' % (opt.save_model, valid_loss, iterations)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir, pattern=r'model_loss_(\d+).(\d+)\_i(\d+).(\d+).pt')
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def run(self, checkpoint=None):

        opt = self.opt
        model_ae = self.model_ae
        optim_ae = self.optim_ae

        if checkpoint is not None:
            self.model_ae.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim_ae.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model_ae, opt)
            resume = False
            start_epoch = 1

        # if we are on a GPU: warm up the memory allocator
        # if self.cuda:
            # self.warm_up()
            #
        valid_loss_ae, valid_loss_lat_dis = self.eval(self.valid_data)
        #
        print('Validation loss ae: %g' % valid_loss_ae)
        print('Validation loss latent discriminator: %g' % valid_loss_lat_dis)
        #
        self.start_time = time.time()

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss_ae, train_loss_lat_dis, train_loss_adv = self.train_epoch(epoch, resume=resume,
                                                                                 itr_progress=itr_progress)

            print('Train loss ae: %g' % train_loss_ae)
            print('Train loss latent discriminator: %g' % train_loss_lat_dis)
            print('Train loss adversarial : %g' % train_loss_adv)

            # #  (2) evaluate on the validation set
            valid_loss_ae, valid_loss_lat_dis = self.eval(self.valid_data)
            print('Validation loss ae: %g' % valid_loss_ae)
            print('Validation loss latent discriminator: %g' % valid_loss_lat_dis)
            #
            self.save(epoch, valid_loss_ae)
            itr_progress = None
            resume = False

    def eval(self, data):
        total_loss_ae = 0
        total_loss_lat_dis = 0
        total_tgt_frames = 0
        total_src = 0
        total_sent = 0
        opt = self.opt

        self.model_ae.eval()
        self.loss_function_ae.eval()
        self.lat_dis.eval()
        self.loss_lat_dis.eval()
        # self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        # print(data_size)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                encoder_outputs, decoder_outputs = self.model_ae(batch)

                gate_padded = batch.get('gate_padded')

                if self.opt.n_frames_per_step > 1:
                    slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1),
                                         self.opt.n_frames_per_step)
                    gate_padded = gate_padded[:, slice]

                src_org = batch.get('source_org')
                src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
                target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
                loss_ae = self.loss_function_ae(decoder_outputs, target)
                loss_ae_data = loss_ae.data.item()

                preds = self.lat_dis(encoder_outputs['context'])

                loss_lat_dis = self.loss_lat_dis(preds, batch.get('source_lang'), mask=encoder_outputs['src_mask'],
                                                 adversarial=False)
                total_src += encoder_outputs['src_mask'].float().sum().item()
                loss_lat_dis_data = loss_lat_dis.data.item()

                total_loss_ae += loss_ae_data
                total_loss_lat_dis += loss_lat_dis_data
                total_tgt_frames += batch.src_size
                total_sent += batch.size
                i = i + 1

        return total_loss_ae / total_tgt_frames, total_loss_lat_dis / total_src * 100

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model_ae.train()
        self.loss_function_ae.train()
        self.lat_dis.train()
        self.loss_lat_dis.train()

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model_ae.zero_grad()
        self.lat_dis.zero_grad()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_loss_ae, total_loss_lat_dis, total_frames, total_loss_adv = 0, 0, 0, 0

        report_loss_ae, report_loss_lat_dis, report_loss_adv, report_tgt_frames, report_sent = 0, 0, 0, 0, 0
        report_dis_frames, report_adv_frames = 0, 0

        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0
        step = 0

        num_accumulated_sents = 0
        grad_scaler = -1

        nan = False
        nan_counter = 0
        n_step_ae = opt.update_frequency
        n_step_lat_dis = opt.update_frequency
        mode_ae = True

        loss_lat_dis = 0.0

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        # while not data_iterator.end_of_epoch():
        while True:

            if data_iterator.end_of_epoch():
                data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                                       epoch=epoch, buffer_size=opt.buffer_size)
                epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)

            batch_size = batch.size
            if grad_scaler == -1:
                grad_scaler = 1  # if self.opt.update_frequency > 1 else batch.tgt_size

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            oom = False
            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                #    targets = batch.get('target_output')
                #  tgt_mask = targets.ne(onmt.constants.PAD)
                if mode_ae:
                    step = self.optim_ae._step
                    loss_ae, loss_adv, encoder_outputs = self.autoencoder_backward(batch, step)
                else:
                    loss_lat_dis, encoder_outputs = self.lat_dis_backward(batch)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                else:
                    raise e

            if loss_ae != loss_ae:
                # catching NAN problem
                oom = True
                self.model_ae.zero_grad()
                self.optim_ae.zero_grad()
                # self.lat_dis.zero_grad()
                # self.optim_lat_dis.zero_grad()

                nan_counter = nan_counter + 1
                print("Warning!!! Loss is Nan")
                if nan_counter >= 15:
                    raise ValueError("Training stopped because of multiple NaN occurence. "
                                     "For ASR, using the Relative Transformer is more stable and recommended.")
            else:
                nan_counter = 0

            if not oom:
                src_size = batch.src_size

                if mode_ae:
                    report_adv_frames += encoder_outputs['src_mask'].sum().item()
                    report_loss_adv += loss_adv
                else:
                    report_dis_frames += encoder_outputs['src_mask'].sum().item()
                    report_loss_lat_dis += loss_lat_dis

                counter = counter + 1

                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True

                if update_flag:
                    # accumulated gradient case, in this case the update frequency
                    if (counter == 1 and self.opt.update_frequency != 1) or counter > 1:
                        grad_denom = 1 / grad_scaler
                    else:
                        grad_denom = 1.0
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    normalize_gradients(amp.master_params(self.optim_ae.optimizer), grad_denom)
                    normalize_gradients(amp.master_params(self.optim_lat_dis.optimizer), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim_ae.optimizer),
                                                       self.opt.max_grad_norm)
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim_lat_dis.optimizer),
                        #                                self.opt.max_grad_norm)

                    torch.nn.utils.clip_grad_value_(amp.master_params(self.optim_lat_dis.optimizer), 0.01)

                    if mode_ae:
                        self.optim_ae.step()
                        self.optim_ae.zero_grad()
                        self.model_ae.zero_grad()
                        self.optim_lat_dis.zero_grad()
                        self.lat_dis.zero_grad()
                    else:
                        self.optim_lat_dis.step()
                        self.optim_lat_dis.zero_grad()
                        self.lat_dis.zero_grad()
                        self.optim_ae.zero_grad()
                        self.model_ae.zero_grad()

                    num_updates = self.optim_ae._step

                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every and mode_ae:
                        # if mode_ae is not here then it will continuously save
                        valid_loss_ae, valid_loss_lat_dis = self.eval(self.valid_data)
                        print('Validation loss ae: %g' % valid_loss_ae)
                        print('Validation loss latent discriminator: %g' % valid_loss_lat_dis)
                        self.save(0, valid_loss_ae, itr=data_iterator)

                    if num_updates == 1000000:
                        break

                    mode_ae = not mode_ae
                    counter = 0
                    # num_accumulated_words = 0

                    grad_scaler = -1
                    num_updates = self.optim_ae._step

                report_loss_ae += loss_ae

                # report_tgt_words += num_words
                num_accumulated_sents += batch_size
                report_sent += batch_size
                total_frames += src_size
                report_tgt_frames += src_size
                total_loss_ae += loss_ae
                total_loss_lat_dis += loss_lat_dis
                total_loss_adv += loss_adv

                optim_ae = self.optim_ae
                optim_lat_dis = self.optim_lat_dis
                # batch_efficiency = total_non_pads / total_tokens

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    log_string = ("loss_ae : %6.2f ;  loss_lat_dis : %6.2f, loss_adv : %6.2f " %
                                  (report_loss_ae / report_tgt_frames,
                                   report_loss_lat_dis / (report_dis_frames + 1e-5),
                                   report_loss_adv / (report_adv_frames + 1e-5)))

                    log_string += ("lr_ae: %.7f ; updates: %7d; " %
                                   (optim_ae.getLearningRate(),
                                    optim_ae._step))

                    log_string += ("lr_lat_dis: %.7f ; updates: %7d; " %
                                   (optim_lat_dis.getLearningRate(),
                                    optim_lat_dis._step))
                    #
                    log_string += ("%5.0f src tok/s " %
                                   (report_tgt_frames / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss_ae = 0
                    report_loss_lat_dis = 0
                    report_loss_adv = 0
                    report_tgt_frames = 0
                    report_dis_frames = 0
                    report_adv_frames = 0
                    report_sent = 0
                    start = time.time()

                i = i + 1

        return total_loss_ae / total_frames * 100, total_loss_lat_dis / n_samples * 100, total_loss_adv / n_samples * 100

