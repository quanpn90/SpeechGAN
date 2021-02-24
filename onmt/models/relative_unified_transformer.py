import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.models.transformer_layers import PositionalEncoding, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.transformers import TransformerEncoder, TransformerDecoder, TransformerDecodingState
import onmt
from onmt.modules.dropout import embedded_dropout
from onmt.models.transformer_layers import XavierLinear, MultiHeadAttention, FeedForward, PrePostProcessing
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer
from onmt.models.relative_transformer_layers import RelativeTransformerEncoderLayer, RelativeTransformerDecoderLayer
from onmt.models.unified_transformer import UnifiedTransformer
from onmt.models.relative_transformer import SinusoidalPositionalEmbedding, LearnablePostionEmbedding, \
    StreamState, StreamDecodingState
from onmt.utils import flip, expected_length
from collections import defaultdict
import math

torch.set_printoptions(profile="full")


def seperate_tensor(input, lengths):

    bsz, tgt_len = input.size(1), input.size(0)

    assert (bsz == 1)

    outputs = list()

    # starting from the first position of the tensor
    offset = 0

    for length in lengths:
        segment = input.narrow(0, offset, length)

        offset += length

        outputs.append(segment)

    return outputs


class RelativeUnifiedTransformer(UnifiedTransformer):
    """
    This class combines the encoder and the decoder into one single sequence
    Joined attention between encoder and decoder parts
    """

    def __init__(self, opt, src_embedding, tgt_embedding, generator, positional_encoder,
                 language_embeddings=None, encoder_type='text', **kwargs):
        self.death_rate = opt.death_rate
        self.bidirectional = opt.bidirectional
        self.layer_modules = []
        self.learnable_position_encoding = opt.learnable_position_encoding
        self.max_memory_size = opt.max_memory_size

        # build_modules will be called from the inherited constructor
        super(RelativeUnifiedTransformer, self).__init__(opt, tgt_embedding, src_embedding,
                                                         generator, positional_encoder,
                                                         language_embeddings=language_embeddings,
                                                         encoder_type=encoder_type)
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        # self.language_embedding = nn.Embedding(3, self.model_size, padding_idx=0)
        self.generator = generator
        self.ignore_source = True
        self.encoder_type = opt.encoder_type

        # learnable position encoding
        if self.learnable_position_encoding:
            self.max_pos_length = opt.max_pos_length
            # pos_emb = self.model_size // self.n_heads
            pos_emb = self.model_size
            self.positional_encoder = LearnablePostionEmbedding(self.max_pos_length, pos_emb)
            print("* Learnable position encoding with max %d positions" % self.max_pos_length)
        else:
            # or using pre-set sinusoidal
            self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        # self.positional_encoder = SinusoidalPositionalEmbedding(opt.model_size)
        self.d_head = self.model_size // self.n_heads

    def gen_mask(self, src, tgt):
        # generate the mask for the mini-batch data

        # both src and tgt are T x B
        input_seq = torch.cat([src, tgt], dim=0)
        seq_len = input_seq.size(0)

        if self.bidirectional:
            bsz, src_len = src.size(1), src.size(0)
            tgt_len = tgt.size(0)

            tgt_tgt_mask = torch.triu(src.new_ones(tgt_len, tgt_len), diagonal=1)
            tgt_src_mask = src.new_zeros(tgt_len, src_len)

            tgt_mask = torch.cat([tgt_src_mask, tgt_tgt_mask], dim=-1)

            src_src_mask = src.new_zeros(src_len, src_len)
            src_tgt_mask = src.new_ones(src_len, tgt_len)

            src_mask = torch.cat([src_src_mask, src_tgt_mask], dim=-1)

            attn_mask = torch.cat([src_mask, tgt_mask], dim=0)

            attn_mask = attn_mask.bool().unsqueeze(-1)

            pad_mask = input_seq.eq(onmt.constants.PAD).unsqueeze(0)

            attn_mask = attn_mask | pad_mask

        else:
            attn_mask = torch.triu(src.new_ones(seq_len, seq_len), diagonal=1).bool().unsqueeze(-1)  # T x T x -1

            pad_mask = input_seq.eq(onmt.constants.PAD).unsqueeze(0)  # 1 x T x B
            # attn_mask = self.mask[:seq_len, :seq_len] + input_seq.eq(onmt.constants.PAD).byte().unsqueeze(1)
            attn_mask = attn_mask | pad_mask

        return attn_mask

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Decoder with Relative Attention with %.2f expected layers" % e_length)

        self.layer_modules = nn.ModuleList()

        for l in range(self.layers):
            # linearly decay the death rate
            death_r = (l + 1.0) / self.layers * self.death_rate

            block = RelativeTransformerDecoderLayer(self.n_heads, self.model_size,
                                                    self.dropout, self.inner_size, self.attn_dropout,
                                                    ignore_source=True,
                                                    variational=self.variational_dropout, death_rate=death_r)
            self.layer_modules.append(block)

    def create_mask_stream(self, src, tgt, src_lengths, tgt_lengths, mem_length=0):

        if self.bidirectional:

            mask = None
            prev_length = 0
            # go through the src and tgt lengths to create mask
            for i, (src_len, tgt_len) in enumerate(zip(src_lengths, tgt_lengths)):

                # print("Step ", i, src_len, tgt_len)
                # first, the source sentence should have full bidirectional attention to the end of itself
                src_mask = src.new_zeros(src_len, src_len + prev_length)

                if prev_length == 0:
                    mask = src_mask
                else:
                    # everything in the past doesn't look at the future

                    prev_mask = src.new_ones(prev_length, src_len)
                    if mask is not None:
                        mask = torch.cat([mask, prev_mask], dim=1)  # prev_len x (src_len + prev_length)
                    else:
                        mask = prev_mask

                    mask = torch.cat([mask, src_mask], dim=0)  # (src_len + prev_length) x (src_len + prev_length)

                prev_length += src_len

                # the target sentence
                # everything in the past doesn't look at the future
                prev_mask = tgt.new_ones(prev_length, tgt_len)

                # the target has unidirectional attention towards everything in the past
                mlen = prev_length
                qlen = tgt_len
                klen = qlen + mlen
                tgt_mask = torch.triu(tgt.new_ones(qlen, klen), diagonal=1 + mlen)

                mask = torch.cat([mask, prev_mask], dim=1)  # prev_len x (prev_len + tgt_len)
                mask = torch.cat([mask, tgt_mask], dim=0)  #

                prev_length += tgt_len

            if mem_length > 0:
                past_mask = src.new_zeros(prev_length, mem_length)
                mask = torch.cat([past_mask, mask], dim=1)

            attn_mask = mask.bool().unsqueeze(-1)

        else:
            seq_len = sum(src_lengths) + sum(tgt_lengths)

            mask = torch.triu(src.new_ones(seq_len, seq_len), diagonal=1)

            if mem_length > 0:
                past_mask = src.new_zeros(seq_len, mem_length)
                mask = torch.cat([past_mask, mask], dim=1)

            attn_mask = mask.bool().unsqueeze(-1)

        return attn_mask

    def forward_stream(self, batch, **kwargs):

        streaming_state = kwargs.get('streaming_state', None)
        src = batch.get('source')  # src_len x batch_size
        tgt = batch.get('target_input')  # (len_tgt x batch_size) x 1

        bsz = src.size(1)
        assert bsz == 1

        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        # First: separate the input tensor into segments
        src_segments = seperate_tensor(src, src_lengths)

        tgt_segments = seperate_tensor(tgt, tgt_lengths)

        # Embedding stage (and scale the embedding)
        embed = self.src_embedding
        if self.word_dropout > 0 and self.training:
            mask = embed.weight.new().resize_((embed.weight.size(0), 1)). \
                       bernoulli_(1 - self.word_dropout).expand_as(embed.weight) / (1 - self.word_dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        # Second: Embedding
        src_embeddings = []
        for src_segment in src_segments:

            src_emb = F.embedding(
                src_segment, masked_embed_weight, padding_idx, embed.max_norm,
                embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

            src_emb.mul_(math.sqrt(self.model_size))

            if self.use_language_embedding:
                if self.language_embedding_type in ["sum", "all_sum"]:
                    src_lang_emb = self.language_embeddings(src_lang)
                    src_emb += src_lang_emb

            src_embeddings.append(src_emb)

        tgt_embeddings = []
        for tgt_segment in tgt_segments:

            tgt_emb = F.embedding(
                tgt_segment, masked_embed_weight, padding_idx, embed.max_norm,
                embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

            tgt_emb.mul_(math.sqrt(self.model_size))

            if self.use_language_embedding:
                if self.language_embedding_type in ["sum", "all_sum"]:
                    tgt_lang_emb = self.language_embeddings(tgt_lang)
                    tgt_emb += tgt_lang_emb

            tgt_embeddings.append(tgt_emb)

        # add src1, tgt1, src2, tgt2 .... srcn, tgtn
        all_embeddings = []
        for (src_emb, tgt_emb) in zip(src_embeddings, tgt_embeddings):
            all_embeddings.append(src_emb)
            all_embeddings.append(tgt_emb)

        emb = torch.cat(all_embeddings, dim=0)

        # prepare attention mask
        mem_length = streaming_state.prev_tgt_mem_size
        attn_mask = self.create_mask_stream(src, tgt, src_lengths, tgt_lengths, mem_length=mem_length)

        klen = emb.size(0) + mem_length

        if self.bidirectional:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)
        else:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = emb

        # Applying dropout
        output = self.preprocess_layer(output)

        pos_emb = self.preprocess_layer(pos_emb)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            buffer = streaming_state.tgt_buffer[i]
            output, coverage, buffer = layer(output, None, pos_emb, attn_mask, None,
                                             incremental=True, incremental_cache=buffer)
            # context and context_mask are None
            streaming_state.tgt_buffer[i] = buffer

        # final layer norm
        output = self.postprocess_layer(output)

        # update the memory and then prune
        streaming_state.prev_tgt_mem_size += klen
        streaming_state.prune_target_memory(self.max_memory_size)

        # now we have to separate the target states from the "output" to generate translations
        target_outputs = []
        contexts = []
        offset = 0
        for (src_len, tgt_len) in zip(src_lengths, tgt_lengths):

            source_output = output.narrow(0, offset, src_len)

            offset += src_len

            target_output = output.narrow(0, offset, tgt_len)

            offset += tgt_len

            target_outputs.append(target_output)
            contexts.append(source_output)

        context = torch.cat(contexts, dim=0)
        output = torch.cat(target_outputs, dim=0)

        output_dict = {'hidden': output, 'coverage': coverage, 'context': context, 'src': src,
                       'target_mask': None}
        output_dict = defaultdict(lambda: None, output_dict)

        # final layer: computing log probabilities
        logprobs = self.generator[0](output_dict)
        output_dict['logprobs'] = logprobs
        output_dict['streaming_state'] = streaming_state

        return output_dict

    def forward(self, batch, target_mask=None, streaming=False, **kwargs):

        if streaming:
            return self.forward_stream(batch, **kwargs)

        src = batch.get('source')  # src_len x batch_size
        tgt = batch.get('target_input')  # len_tgt x batch_size
        src_pos = batch.get('source_pos')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        tgt_len = tgt.size(0)
        src_len = src.size(0)
        bsz = tgt.size(1)

        # Embedding stage (and scale the embedding)
        embed = self.src_embedding
        if self.word_dropout > 0 and self.training:
            mask = embed.weight.new().resize_((embed.weight.size(0), 1)). \
                       bernoulli_(1 - self.word_dropout).expand_as(embed.weight) / (1 - self.word_dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        src_emb = F.embedding(
            src, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        src_emb.mul_(math.sqrt(self.model_size))

        tgt_emb = F.embedding(
            tgt, masked_embed_weight, padding_idx, embed.max_norm,
            embed.norm_type, embed.scale_grad_by_freq, embed.sparse)

        tgt_emb.mul_(math.sqrt(self.model_size))

        if self.use_language_embedding:
            if self.language_embedding_type in ["sum", "all_sum"]:
                src_lang_emb = self.language_embeddings(src_lang)
                src_emb += src_lang_emb
                tgt_lang_emb = self.language_embeddings(tgt_lang)
                tgt_emb += tgt_lang_emb
            else:
                raise NotImplementedError

        # concatenate embedding
        emb = torch.cat([src_emb, tgt_emb], dim=0)  # L x batch_size x H

        # prepare self-attention mask
        attn_mask = self.gen_mask(src, tgt)

        # pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        klen = src_len + tgt_len

        if self.bidirectional:
            pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)
        else:
            pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = emb

        # Applying dropout
        output = self.preprocess_layer(output)

        pos_emb = self.preprocess_layer(pos_emb)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            output, coverage, _ = layer(output, None, pos_emb, attn_mask, None)  # context and context_mask are None

        # Final normalization
        output = self.postprocess_layer(output)

        # extract the "source" and "target" parts of the output
        context = output[:src_len, :, :]
        output = output[-tgt_len:, :, :]
        output_dict = {'hidden': output, 'coverage': coverage, 'context': context, 'src': src,
                       'target_mask': target_mask}

        # final layer: computing log probabilities
        logprobs = self.generator[0](output_dict)
        output_dict['logprobs'] = logprobs

        return output_dict

    def encode(self, input, decoder_state, input_pos=None, input_lang=None):

        buffers = decoder_state.attention_buffers
        src_lang = input_lang
        input = input.transpose(0, 1)
        # Embedding stage (and scale the embedding)
        src_emb = embedded_dropout(self.src_embedding, input, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)

        if self.use_language_embedding:
            if self.language_embedding_type in ["sum", "all_sum"]:
                src_lang_emb = self.language_embeddings(src_lang)
                src_emb += src_lang_emb

        emb = src_emb
        src_len = input.size(0)
        bsz = input.size(1)
        mask_src_src = input.eq(onmt.constants.PAD).byte()  # B x 1 x src_len
        mask_src = mask_src_src.unsqueeze(0)

        attn_mask = mask_src.bool()  # L x L x batch_size

        output = emb

        # Applying dropout and tranpose to T x B x H
        output = self.preprocess_layer(output)

        klen = src_len
        pos = torch.arange(klen - 1, -klen, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            # context and context_mask are None
            buffer = buffers[i] if i in buffers else None
            # output, coverage, buffer = layer.step(output, None, attn_mask, None, buffer)
            output, coverage, buffer = layer(output, None, pos_emb, attn_mask, None,
                                             incremental=True, incremental_cache=buffer)
            decoder_state.update_attention_buffer(buffer, i)

        # Final normalization
        output = self.postprocess_layer(output)

        return output, decoder_state

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """
        # raise NotImplementedError
        tgt_output = batch.get('target_output')
        output_dict = self.forward(batch, target_mask=None)
        context = output_dict['context']
        logprobs = output_dict['logprobs']

        batch_size = logprobs.size(1)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()

        for gen_t, tgt_t in zip(logprobs, tgt_output):
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):

        # This model uses pre-allocated position encoding
        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len + 1, new_len + 1)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

        return

    def reset_states(self):
        return

    def step(self, input, decoder_state):

        src = decoder_state.src if decoder_state.src is not None else None
        tgt = input.transpose(0, 1)
        tgt_lang = decoder_state.tgt_lang
        src_lang = decoder_state.src_lang
        buffers = decoder_state.attention_buffers

        tgt_len = tgt.size(0)
        src_len = src.size(0)
        bsz = tgt.size(1)

        # Embedding stage (and scale the embedding)
        # src_emb = embedded_dropout(self.src_embedding, src, dropout=self.word_dropout if self.training else 0) \
        #           * math.sqrt(self.model_size)
        input_ = tgt[-1:]
        tgt_emb = embedded_dropout(self.tgt_embedding, input_, dropout=self.word_dropout if self.training else 0) \
                  * math.sqrt(self.model_size)

        if self.use_language_embedding:
            if self.language_embedding_type in ["sum", "all_sum"]:
                # src_lang_emb = self.language_embeddings(src_lang)
                # src_emb += src_lang_emb
                tgt_lang_emb = self.language_embeddings(tgt_lang)
                tgt_emb += tgt_lang_emb
            else:
                raise NotImplementedError

        # concatenate embedding
        # emb = torch.cat([src_emb, tgt_emb], dim=0)  # L x batch_size x H
        emb = tgt_emb

        # prepare self-attention mask

        attn_mask = self.gen_mask(src, tgt)
        # last attn_mask step
        attn_mask = attn_mask[-1:, :, :]

        klen = src_len + tgt_len
        pos = torch.arange(klen - 1, -1, -1.0, device=emb.device, dtype=emb.dtype)

        pos_emb = self.positional_encoder(pos)

        output = emb

        # Applying dropout
        output = self.preprocess_layer(output)

        # FORWARD PASS
        coverage = None
        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            output, coverage, buffer = layer(output, None, pos_emb, attn_mask, None,
                                             incremental=True,
                                             incremental_cache=buffer)  # context and context_mask are None
            decoder_state.update_attention_buffer(buffer, i)

        # Final normalization
        output = self.postprocess_layer(output)

        # output = output[-1:, :, :]

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output

        logprobs = self.generator[0](output_dict).squeeze(0)

        output_dict['src'] = decoder_state.src.transpose(0, 1)
        output_dict['log_prob'] = logprobs
        output_dict['coverage'] = logprobs.new(bsz, tgt_len, src_len).zero_()

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1):

        src = batch.get('source')
        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)  # B x T

        decoder_state = TransformerDecodingState(src, tgt_lang, None, None,
                                                 beam_size=beam_size, model_size=self.model_size, type=type)

        # forward pass through the input to get the buffer
        # src_transposed = src_transposed.repeat(beam_size, 1)
        encoder_output, decoder_state = self.encode(src_transposed, decoder_state, input_pos=src_pos,
                                                    input_lang=src_lang)

        decoder_state.src_lang = src_lang

        buffers = decoder_state.attention_buffers
        bsz = src.size(1)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src.device)

        for l in buffers:
            buffer_ = buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, new_order)  # 1 for time first

        return decoder_state

    def tie_weights(self):
        assert self.generator is not None, "The generator needs to be created before sharing weights"
        self.generator[0].linear.weight = self.tgt_embedding.weight

    def share_enc_dec_embedding(self):
        self.src_embedding.weight = self.tgt_embedding.weight

    def init_stream(self):

        param = next(self.parameters())
        layers = self.layers
        streaming_state = StreamState(layers, self.max_memory_size, param.device, param.dtype)
        return streaming_state
