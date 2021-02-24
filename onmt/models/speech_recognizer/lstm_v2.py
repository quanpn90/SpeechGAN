import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
import math
import onmt
from onmt.modules.base_seq2seq import NMTModel, DecoderState
from onmt.models.transformer_layers import PrePostProcessing
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.optimized.encdec_attention import EncdecMultiheadAttn
from onmt.modules.dropout import embedded_dropout
from .relative_transformer_layers import LIDFeedForward_small

# from . import WordDropout, freeze_module
# from .attn import MultiHeadAttention


class SpeechLSTMEncoder(nn.Module):
    def __init__(self, opt, embedding, encoder_type='audio'):
        super(SpeechLSTMEncoder, self).__init__()
        self.opt = opt
        self.model_size = opt.model_size

        if hasattr(opt, 'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers

        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout

        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling

        self.switchout = opt.switchout
        self.varitional_dropout = opt.variational_dropout
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        self.time = opt.time
        self.lsh_src_attention = opt.lsh_src_attention
        self.reversible = opt.src_reversible

        self.temperature = math.sqrt(opt.model_size)

        if self.switchout > 0.0:
            self.word_dropout = 0.0

        feature_size = opt.input_size
        self.channels = 1

        self.lid_network = None

        if opt.gumbel_embedding or opt.lid_loss or opt.bottleneck:
            self.lid_network = LIDFeedForward_small(opt.model_size, opt.model_size, opt.n_languages, dropout=opt.dropout)
            self.n_languages = opt.n_languages

        if opt.upsampling:
            feature_size = feature_size // 4

        if not self.cnn_downsampling:
            self.audio_trans = nn.Linear(feature_size, self.model_size)
            torch.nn.init.xavier_uniform_(self.audio_trans.weight)
        else:
            channels = self.channels
            cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
                   nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]

            feat_size = (((feature_size // channels) - 3) // 4) * 32
            # cnn.append()
            self.audio_trans = nn.Sequential(*cnn)
            self.linear_trans = nn.Linear(feat_size, self.model_size)
            # assert self.model_size == feat_size, \
            #     "The model dimension doesn't match with the feature dim, expecting %d " % feat_size

        # if use_cnn:
        #     cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)),
        #            nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std))]
        #     self.cnn = nn.Sequential(*cnn)
        #     input_size = ((((input_size - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1) * 32
        # else:
        #     self.cnn = None

        self.unidirect = self.opt.unidirectional

        self.rnn = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
                           bidirectional=(not self.unidirect), bias=False, dropout=self.dropout, batch_first=True)

        # self.rnn_1 = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
        #                      bias=False, dropout=self.dropout, batch_first=True)
        #
        # self.rnn_2 = nn.LSTM(input_size=self.model_size, hidden_size=self.model_size, num_layers=self.layers,
        #                      bias=False, dropout=self.dropout, batch_first=True)
        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.varitional_dropout)
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        # self.incl_win = opt.incl_win

    # TO_DO
    # def rnn_fwd_2(self, seq, mask, hid):
    #     if mask is not None:
    #         seq_1 = self.rnn_1(seq) * mask.unsqueeze(-1)
    #         seq_inverse = torch.flip(seq, [1])
    #         seq_2 = self.rnn_2(seq_inverse) * torch.flip(mask, [1]).unsqueeze(-1)
    #
    #     else:
    #         seq_1 = self.rnn_1(seq)
    #         seq_2 = self.rnn_2(torch.flip(seq, [1]))
    #
    #     return seq_1 + seq_2

    # def rnn_fwd_incl(self, seq, mask, hid=None):
    #     win, time_len = self.incl_win, seq.size(1)
    #     out = []
    #     for i in range((time_len - 1) // win + 1):
    #         s, e = win * i, min(time_len, win * (i + 1))
    #         src = seq[:, s:e, :]
    #         enc, hid = self.rnn(src, hid)
    #         out.append(enc)
    #     out = torch.cat(out, dim=1)
    #     # out *= mask.unsqueeze(-1).type(out.dtype)
    #
    #     return out, hid
    def rnn_fwd(self, seq, mask, hid):
        if mask is not None:
            lengths = mask.sum(-1).float()
            seq = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
            seq, hid = self.rnn(seq, hid)
            seq = pad_packed_sequence(seq, batch_first=True)[0]
        else:
            seq, hid = self.rnn(seq)

        return seq, hid

    # def rnn_fwd_2(self, seq, mask, hid):
    #     # print(mask)
    #     # print(mask.shape)
    #     if mask is not None:
    #         # print(mask.shape)
    #         lengths = mask.sum(-1).float()
    #         lengths_2 = torch.ones(mask.shape[0]) * torch.max(lengths)
    #         seq_2 = pack_padded_sequence(seq, lengths_2, batch_first=True, enforce_sorted=False)
    #
    #         seq_2, hid = self.rnn(seq_2, hid)
    #         seq_2 = pad_packed_sequence(seq_2, batch_first=True)[0]
    #
    #     else:
    #         seq_2, hid = self.rnn(seq)
    #
    #     return seq_2, hid

    def forward(self, input, hid=None):
        # print(input)
        if not self.cnn_downsampling:
            mask_src = input.narrow(2, 0, 1).squeeze(2).gt(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)
            emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                    input.size(1), -1)
            emb = emb.type_as(input)
        else:

            long_mask = input.narrow(2, 0, 1).squeeze(2).gt(onmt.constants.PAD)
            input = input.narrow(2, 1, input.size(2) - 1)
            # first resizing to fit the CNN format
            input = input.view(input.size(0), input.size(1), -1, self.channels)
            input = input.permute(0, 3, 1, 2)

            input = self.audio_trans(input)
            input = input.permute(0, 2, 1, 3).contiguous()
            input = input.view(input.size(0), input.size(1), -1)
            input = self.linear_trans(input)

            mask_src = long_mask[:, 0:input.size(1) * 4:4]
            # the size seems to be B x T ?
            emb = input
        seq, hid = self.rnn_fwd(emb, mask_src, hid)
        # print("time_1  " + str(time.time() - start ))

        if not self.unidirect:
            hidden_size = seq.size(2) // 2
            seq = seq[:, :, :hidden_size] + seq[:, :, hidden_size:]

        # print(seq.shape)
        #
        # print(emb.shape)
        seq = self.postprocess_layer(seq)

        lid_vector = None
        lid_logits = None
        bottleneck = None
        if self.lid_network is not None:

            avg_seq = torch.mean(seq, dim=1)
            lid_logits, bottleneck = self.lid_network(avg_seq)
            #,tau=math.sqrt(self.n_languages)
            temperature = 0.01
            lid_logits_temp = lid_logits / temperature
            lid_vector = F.softmax(lid_logits_temp, dim=-1)


        output_dict = {'context': seq.transpose(0, 1), 'src_mask': mask_src , 'lid_vector': lid_vector , 'lid_logits':lid_logits, 'bottleneck' : bottleneck}

        return output_dict


class SpeechLSTMDecoder(nn.Module):
    def __init__(self, opt, embedding, language_embeddings=None, ignore_source=False, allocate_positions=True):
        super(SpeechLSTMDecoder, self).__init__()

        # Keep for reference

        # Define layers
        self.model_size = opt.model_size
        self.layers = opt.layers
        self.dropout = opt.dropout

        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.variational_dropout = opt.variational_dropout

        self.encoder_type = opt.encoder_type

        self.lstm = nn.LSTM(self.model_size, self.model_size, self.layers, dropout=self.dropout, batch_first=True)

        self.fast_self_attention = opt.fast_self_attention

        if opt.fast_xattention:
            self.multihead_tgt =  EncdecMultiheadAttn(opt.n_heads, opt.model_size, opt.attn_dropout)
        else:
            self.multihead_tgt = MultiHeadAttention(opt.n_heads, opt.model_size, attn_p=opt.attn_dropout, share=3)

        # self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
        #                                           variational=self.variational_dropout)
        self.preprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')
        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        self.encoder_cnn_downsampling = opt.cnn_downsampling
        self.language_embeddings = language_embeddings
        self.use_language_embedding = opt.use_language_embedding
        self.gumbel_embedding = opt.gumbel_embedding
        self.bottleneck = opt.bottleneck
        self.language_embedding_type = opt.language_embedding_type

        if self.language_embedding_type == 'concat':
            self.projector = nn.Linear(opt.model_size * 2, opt.model_size)

    def process_embedding(self, input, input_lang=None):

        return input

    def step(self, input, decoder_state, **kwargs):

        context = decoder_state.context
        bottleneck = decoder_state.bottleneck
        lid_vector = decoder_state.lid_vector

        buffer = decoder_state.LSTM_buffer
        attn_buffer = decoder_state.attention_buffers
        hid = buffer["hidden_state"]
        cell = buffer["cell_state"]
        buffering = decoder_state.buffering
        if hid is not None:
            hid_cell = (hid, cell)
        else:
            hid_cell = None

        lang = decoder_state.tgt_lang

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)

        src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        if input.size(1) > 1:
            input_ = input[:, -1].unsqueeze(1)
        else:
            input_ = input

        emb = self.word_lut(input_)

        emb = emb * math.sqrt(self.model_size)

        if self.use_language_embedding:
            # print("Using language embedding")
            if lid_vector is not None and self.gumbel_embedding:

                lang_emb = lid_vector.matmul(self.language_embeddings.weight)

            elif bottleneck is not None and self.bottleneck:
                lang_emb = bottleneck
            else:
                lang_emb = self.language_embeddings(lang)  # B x H or 1 x H

            if self.language_embedding_type == 'sum':
                print(emb.shape)
                print(lang_emb.shape)
                dec_emb = emb + lang_emb.unsqueeze(1)
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[0])
                emb[0] = bos_emb

                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                dec_emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        if context is not None:
            if self.encoder_type == "audio":
                if src.data.dim() == 3:
                    if self.encoder_cnn_downsampling:
                        long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                elif self.encoder_cnn_downsampling:
                    long_mask = src.eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
            else:
                mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        if input_.size(0) > 1 and input_.size(1) > 1:
            # print(dec_seq)
            #
            lengths = input.gt(onmt.constants.PAD).sum(-1)

            dec_in = pack_padded_sequence(dec_emb, lengths, batch_first=True, enforce_sorted=False)
            dec_out, hidden = self.lstm(dec_in, hid_cell)
            dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        else:

            dec_out, hid_cell = self.lstm(dec_emb, hid_cell)

        decoder_state.update_LSTM_buffer(hid_cell)

        lt = input_.size(1)
        attn_mask = mask_src.expand(-1, lt, -1)
        dec_out = self.preprocess_layer(dec_out)

        dec_out = dec_out.transpose(0, 1)
        if buffering:
            buffer = attn_buffer[0]
            if buffer is None:
                buffer = dict()

            output, coverage = self.multihead_tgt(dec_out, context, context, attn_mask,
                                                  incremental=True, incremental_cache=buffer)
            decoder_state.update_attention_buffer(buffer, 0)
        else:
            output, coverage = self.multihead_tgt(dec_out, context, context, attn_mask)
        output = (output + dec_out)
        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': context})

        return output_dict

    def forward(self, dec_seq, enc_out, src, tgt_lang=None, hid=None, lid_vector=None, bottleneck = None, **kwargs):

        emb = embedded_dropout(self.word_lut, dec_seq, dropout=self.word_dropout if self.training else 0)
        emb = emb * math.sqrt(self.model_size)

        if self.use_language_embedding:
            # print("Using language embedding")
            if lid_vector is not None and self.gumbel_embedding:

                lang_emb = lid_vector.matmul(self.language_embeddings.weight)

            elif bottleneck is not None and self.bottleneck:
                lang_emb = bottleneck
            else:
                lang_emb = self.language_embeddings(tgt_lang)  # B x H or 1 x H

            if self.language_embedding_type == 'sum':

                dec_emb = emb + lang_emb.unsqueeze(1)
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[0])
                emb[0] = bos_emb

                lang_emb = lang_emb.unsqueeze(0).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                dec_emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError

        if enc_out is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0: enc_out.size(0) * 4:4].unsqueeze(1)
                    # print("long_mask.shape")
                    # print(mask_src.shape)
            else:

                mask_src = src.data.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        # dec_emb = self.preprocess_layer(dec_emb.transpose(0, 1).contiguous())

        if dec_seq.size(0) > 1 and dec_seq.size(1) > 1:
            # print(dec_seq)
            #
            lengths = dec_seq.gt(onmt.constants.PAD).sum(-1)

            # print(lengths)
            dec_in = pack_padded_sequence(dec_emb, lengths, batch_first=True, enforce_sorted=False)
            dec_out, hid = self.lstm(dec_in, hid)
            dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        else:

            dec_out, hid = self.lstm(dec_emb, hid)

        lt = dec_seq.size(1)
        # print("dec_out")
        # print(dec_out.shape)
        attn_mask = mask_src.expand(-1, lt, -1)
        dec_out = self.preprocess_layer(dec_out)

        dec_out = dec_out.transpose(0, 1).contiguous()
        enc_out = enc_out.contiguous()
        # enc_out = enc_out.transpose(0,1)
        output, coverage = self.multihead_tgt(dec_out, enc_out, enc_out, attn_mask)
        output = (output + dec_out)

        output = self.postprocess_layer(output)
        # output = self.project(context)
        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': enc_out})
        return output_dict


class SpeechLSTMSeq2Seq(NMTModel):
    def __init__(self, encoder, decoder, generator=None, rec_decoder=None, rec_generator=None, mirror=False):
        super().__init__(encoder, decoder, generator, rec_decoder, rec_generator)

        self.model_size = self.decoder.model_size
        self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)
        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0


    def tie_weights_lid(self):
        assert self.encoder.lid_network is not None, "The Lid network needs to be created before sharing weights"
        self.encoder.lid_network.last_linear.weight = self.decoder.language_embeddings.weight

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None, nce=False):
        src = batch.get('source')
        tgt = batch.get('target_input')
        src_pos = batch.get('source_pos')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        org_src = src
        org_tgt = tgt
        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        encoder_output = self.encoder(src)
        encoder_output = defaultdict(lambda: None, encoder_output)

        context = encoder_output['context']
        src_mask = encoder_output['src_mask']
        lid_vector = encoder_output['lid_vector']
        bottleneck = encoder_output['bottleneck']

        decoder_output = self.decoder(tgt, context, src,
                                      tgt_lang=tgt_lang, lid_vector=lid_vector, bottleneck=bottleneck, input_pos=tgt_pos, streaming=streaming,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                      streaming_state=streaming_state)

        decoder_output = defaultdict(lambda: None, decoder_output)
        # streaming_state = decoder_output['streaming_state']
        output = decoder_output['hidden']

        output_dict = defaultdict(lambda: None, decoder_output)
        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['reconstruct'] = None
        # output_dict['streaming_state'] = streaming_state
        output_dict['target'] = batch.get('target_output')
        output_dict['lid_logits'] = encoder_output['lid_logits']
        if self.training and nce:
            output_dict = self.generator[0](output_dict)
        else:
            logprobs = self.generator[0](output_dict)['logits']
            output_dict['logprobs'] = logprobs

        return output_dict

    def step(self, input_t, decoder_state):

        output_dict = self.decoder.step(input_t, decoder_state)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        # squeeze to remove the time step dimension
        log_prob = self.generator[0](output_dict)['logits'].squeeze(0)
        # print("log_prob.shape")
        # print(log_prob.shape)
        log_prob = F.log_softmax(log_prob, dim=-1, dtype=torch.float32)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param buffering:
        :param streaming:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)

        encoder_output = self.encoder(src_transposed)
        decoder_state = LSTMDecodingState(src, tgt_lang, encoder_output['context'], encoder_output['lid_vector'] ,encoder_output['bottleneck'],
                                beam_size=beam_size, model_size=self.model_size,
                                type=type, buffering=buffering)

        return decoder_state

    def decode(self, batch):
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_pos = batch.get('target_pos')
        # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        encoder_output = self.encoder(src)

        context = encoder_output['context']
        lid_vector = encoder_output['lid_vector']
        bottleneck = encoder_output['bottleneck']
        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        decoder_output = self.decoder(tgt_input, context, src, tgt_lang=tgt_lang, lid_vector=lid_vector, bottleneck=bottleneck, src_lang=src_lang,
                                      input_pos=tgt_pos)['hidden']

        output = decoder_output

        for dec_t, tgt_t in zip(output, tgt_output):

            dec_out = defaultdict(lambda: None)
            dec_out['hidden'] = dec_t.unsqueeze(0)
            dec_out['src'] = src
            dec_out['context'] = context

            if isinstance(self.generator, nn.ModuleList):
                gen_t = self.generator[0](dec_out)['logits']
            else:
                gen_t = self.generator(dec_out)['logits']
            gen_t = F.log_softmax(gen_t, dim=-1, dtype=torch.float32)
            gen_t = gen_t.squeeze(0)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores


class LSTMDecodingState(DecoderState):
    def __init__(self, src, tgt_lang, context, lid_vector = None, bottleneck = None, beam_size=1, model_size=512, type=2,
                 cloning=True, buffering=False):
        self.beam_size = beam_size
        self.model_size = model_size
        self.LSTM_buffer = dict()
        self.LSTM_buffer["hidden_state"] = None
        self.LSTM_buffer["cell_state"] = None
        self.buffering = buffering
        self.attention_buffers = defaultdict(lambda: None)
        self.lid_vector = lid_vector.repeat(beam_size, 1)
        self.bottleneck = bottleneck.repeat(beam_size, 1)
        if type == 1:
            # if audio only take one dimension since only used for mask
            # raise NotImplementedError
            self.original_src = src  # TxBxC
            self.concat_input_seq = True

            if src is not None:
                if src.dim() == 3:
                    self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
                    # self.src = src.repeat(1, beam_size, 1) # T x Bb x c
                else:
                    self.src = src.repeat(1, beam_size)
            else:
                self.src = None

            if context is not None:
                self.context = context.repeat(1, beam_size, 1)
            else:
                self.context = None

            self.input_seq = None
            # self.src_mask = None
            self.tgt_lang = tgt_lang

        elif type == 2:
            bsz = src.size(1)  # src is T x B
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(src.device)

            if cloning:
                self.src = src.index_select(1, new_order)  # because src is batch first

                if context is not None:
                    self.context = context.index_select(1, new_order)
                else:
                    self.context = None

                # if src_mask is not None:
                #     self.src_mask = src_mask.index_select(0, new_order)
                # else:
                #     self.src_mask = None
            else:
                self.context = context
                self.src = src
                # self.src_mask = src_mask
            self.input_seq = None
            self.concat_input_seq = False
            self.tgt_lang = tgt_lang
        else:
            raise NotImplementedError


    def update_LSTM_buffer(self, buffer):

        hid, cell = buffer
        self.LSTM_buffer["hidden_state"] = hid
        self.LSTM_buffer["cell_state"] = cell

    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer

    def update_beam(self, beam, b, remaining_sents, idx):

        if self.beam_size == 1:
            return
        # print(self.input_seq)
        # print(self.src.shape)
        for tensor in [self.src, self.input_seq, self.lid_vector, self.bottleneck]:

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(1, beam[b].getCurrentOrigin()))
            # print(sent_states.shape)
            # print(self.src.shape)
        # for l in self.LSTM_buffers:
        for l in self.LSTM_buffer:
            buffer_ = self.LSTM_buffer[l]

            t_, br_, d_ = buffer_.size()
            # print(buffer_.size())
            sent_states = buffer_.view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

            sent_states.data.copy_(sent_states.data.index_select(1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffers = self.attention_buffers[l]
            if buffers is not None:
                for k in buffers.keys():
                    buffer_ = buffers[k]
                    t_, br_, d_ = buffer_.size()
                    sent_states = buffer_.view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                    sent_states.data.copy_(sent_states.data.index_select(1, beam[b].getCurrentOrigin()))

    def prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.model_size

        def update_active_with_hidden(t):
            if t is None:
                return t
            dim = t.size(-1)
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_without_hidden(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            return new_t

        self.context = update_active_with_hidden(self.context)

        self.input_seq = update_active_without_hidden(self.input_seq)

        if self.src.dim() == 2:
            self.src = update_active_without_hidden(self.src)
        elif self.src.dim() == 3:
            t = self.src
            dim = t.size(-1)
            view = t.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            self.src = new_t

        for l in self.LSTM_buffer:
            buffer_ = self.LSTM_buffer[l]

            buffer = update_active_with_hidden(buffer_)

            self.LSTM_buffer[l] = buffer

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    buffer_[k] = update_active_with_hidden(buffer_[k])

    def _reorder_incremental_state(self, reorder_state):

        if self.context is not None:
            self.context = self.context.index_select(1, reorder_state)

        # if self.src_mask is not None:
        #     self.src_mask = self.src_mask.index_select(0, reorder_state)
        self.src = self.src.index_select(1, reorder_state)

        self.lid_vector = self.lid_vector.index_select(0, reorder_state)
        self.bottleneck = self.bottleneck.index_select(0, reorder_state)
        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state)  # 1 for time first

        for l in self.LSTM_buffer:
            buffer_ = self.LSTM_buffer[l]
            if buffer_ is not None:
                self.LSTM_buffer[l] = buffer_.index_select(1, reorder_state)



