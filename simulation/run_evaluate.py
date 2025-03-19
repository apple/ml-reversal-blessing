#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from omegaconf import DictConfig, OmegaConf
import argparse
from typing import List, Tuple, Union
from tqdm import tqdm

import torch
import torch.nn.functional as F
import transformers

from lm_eval.tasks import get_task_dict, TaskManager
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    Collator,
    pad_and_concat,
)
from simulation.evaluation import evaluate

from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.recipe_interfaces import EvalRecipeInterface
from torchtune import config, training, utils
from torchtune.modules import TransformerDecoder
from torchtune.data import (
    left_pad_sequence,
)
from torchtune.generation import generate


class EleutherEvalRecipe(EvalRecipeInterface):
    """
    This recipe runs evaluation on a trained model using EleutherAI's eval harness.
    This assumes the user has the EleutherAI eval harness installed. See
    https://github.com/EleutherAI/lm-evaluation-harness for more details.

    Features:
        - Single GPU evaluation. Multi-GPU evaluation is currently not supported.
        - Quantization (for text-only models) is supported.
        - Any task from the EleutherAI eval harness

    We recommend launching evaluation using the tune CLI::

        tune run eleuther_eval --config eleuther_evaluation \
            tasks=["truthfulqa_mc2","hellaswag"] \
            limit=50 \
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.device = utils.get_device(device=cfg.device)
        self.dtype = training.get_dtype(dtype=cfg.dtype, device=self.device)
        self.logger = utils.get_logger("info")
        training.set_seed(seed=cfg.seed)

        # Eval specific variables
        self.limit = cfg.limit
        self.tasks = list(cfg.tasks)
        self.batch_size = cfg.batch_size
        self.enable_kv_cache = True
        self.include_path = None
        self._cfg = cfg

    def setup(self, cfg: DictConfig) -> None:
        # Load checkpoint
        checkpointer = config.instantiate(cfg.checkpointer)

        # Initialize model
        with training.set_default_dtype(self.dtype), self.device:
            model = config.instantiate(cfg.model)

        ckpt_dict = checkpointer.load_checkpoint()[training.MODEL_KEY]
        model.load_state_dict(ckpt_dict)

        # Load model weights into initialized model
        self.logger.info(f"Model is initialized with precision {self.dtype}.")

        model.eval()

        # Initialize tokenizer/transform
        model_transform = config.instantiate(cfg.tokenizer)

        if isinstance(model, TransformerDecoder):
            eleuther_model_wrapper = _LLMEvalWrapper
            if self._cfg.reverse_eval:
                eleuther_model_wrapper = _EvalWrapperReversed
            else:
                eleuther_model_wrapper = _LLMEvalWrapper
        self.eleuther_model_wrapper = eleuther_model_wrapper(
            model,
            model_transform,
            device=self.device,
            max_seq_length=cfg.max_seq_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            enable_kv_cache=self.enable_kv_cache,
        )
        self.eleuther_model_wrapper_r2l = _EvalWrapperReversed(
            model,
            model_transform,
            device=self.device,
            max_seq_length=cfg.max_seq_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            enable_kv_cache=self.enable_kv_cache,
        )


class _LLMEvalWrapper(HFLM):
    """An EvalWrapper for EleutherAI's eval harness based on gpt-fast's
    EvalWrapper: https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py.

    Note:
        This is for text-only decoder models.

    Args:
        model (TransformerDecoder): The model to evaluate.
        tokenizer (ModelTokenizer): Tokenizer associated with the model being evaluated.
            This should be the same tokenizer used when fine-tuning the model.
        device (torch.device): The device to use.
        max_seq_length (int): The maximum sequence length to use.
        batch_size (int): The batch size per GPU to use.
        dtype (torch.dtype): dtype for the model caches during generation.
        enable_kv_cache (bool): Whether to enable KV cache for generation.
    """

    def __init__(
        self,
        model: TransformerDecoder,
        tokenizer: ModelTokenizer,
        *,
        device: torch.device,
        max_seq_length: int = 4096,
        batch_size: int = 8,
        dtype: torch.dtype = torch.float32,
        enable_kv_cache: bool = True,
    ):
        # TODO (@joecummings): Remove this init function so we don't load in extraneous stuff
        super().__init__(pretrained="gpt2", device=str(device))
        self._model = model
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size
        self._dtype = dtype
        self._enable_kv_cache = enable_kv_cache

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def enable_kv_cache(self):
        return self._enable_kv_cache

    def tok_encode(self, text: str, **kwargs) -> List[int]:
        # Note on add_bos flag: setting to False as this gives better results, for example
        # +1% on truthfulqa_mc2 with a LoRA finetune. lit-gpt also sets this to False,
        # see https://github.com/Lightning-AI/lit-gpt/blob/main/eval/lm_eval_harness.py#L66,
        # though notably fast-gpt does the opposite
        # https://github.com/pytorch-labs/gpt-fast/blob/main/eval.py#L123.
        return self._tokenizer.encode(text=text, add_bos=True, add_eos=False)

    def tok_batch_encode(
        self, text: List[str], left_truncate_len: int = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_text = [self.tok_encode(x) for x in text]

        # pad left
        x = left_pad_sequence(
            [torch.tensor(x) for x in tokenized_text],
            batch_first=True,
            padding_value=self._tokenizer.pad_id,
        )

        # the harness will use left_truncate_len to indicate that the current batch
        # needs to be truncated to self.max_seq_len - self.max_gen_toks
        if left_truncate_len is not None:
            x = x[:, -left_truncate_len:]

        return x, torch.ones_like(x)  # return 'mask' b/c it's expected by the harness

    def tok_decode(self, tokens: Union[List[int], int], **kwargs) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._model(inps)

    @torch.inference_mode()
    def _model_generate(
        self, context: torch.Tensor, **generation_kwargs
    ) -> torch.Tensor:
        bsz, seq_len = context.shape

        temperature = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", False)
        if do_sample or temperature != 0.0:
            raise RuntimeError(
                "Any decoding strategy other than greedy is not supported."
            )

        # Setup KV caches OR reset them if they're already set up
        if self.enable_kv_cache:
            if self.model.caches_are_enabled():
                self.model.reset_caches()
            else:
                with self.device:
                    self.model.setup_caches(
                        batch_size=self.batch_size,
                        dtype=self._dtype,
                        decoder_max_seq_len=self.max_length,
                    )

        # if we've recieved fewer than self._batch_size samples in the current
        # batch we need to pad the batch out. here we're padding the end of the
        # current batch to the correct length. this is because when we use static
        # KV-caches, the model will expect a fixed batch size for all samples.
        maybe_padded_context = torch.nn.functional.pad(
            context,
            (0, 0, 0, self._batch_size - bsz),
            value=self._tokenizer.eos_id,  # pad with one of the tokenizer's stop tokens so generation can stop early
        )

        toks, _ = generate(
            self.model,
            maybe_padded_context,
            max_generated_tokens=self.max_gen_toks,
            temperature=temperature,
            top_k=None,
            stop_tokens=self._tokenizer.stop_tokens,
        )
        return toks[:bsz]

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=(
                "contexts"
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                and self.logits_cache
                else None
            ),
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]
            count = 0
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                assert (
                    logits.shape[0] == padding_len_inp
                ), "I don't see these two would be different in any case"
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )
                if inplen - contlen != 0:
                    ctx_logits = logits[: (inplen - contlen)]
                    ctx_label = torch.tensor(ctx_tokens[1:], device=ctx_logits.device)
                    ctx_loss = ce_loss_fn(ctx_logits, ctx_label)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):

                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    cont_loss = ce_loss_fn(logits.squeeze(0), cont_toks.squeeze(0))
                    if inplen - contlen == 0:
                        full_loss = cont_loss
                        ctx_loss = cont_loss
                    else:
                        full_loss = torch.cat([cont_loss, ctx_loss])
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (
                        (
                            torch.clamp(ctx_loss, max=10).tolist(),
                            torch.clamp(cont_loss, max=10).tolist(),
                            torch.clamp(full_loss, max=10).tolist(),
                        ),
                        float(logits.sum()),
                        bool(max_equal),
                    )

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)


class _EvalWrapperReversed(_LLMEvalWrapper):
    def tok_encode(self, text: str, **kwargs) -> List[int]:
        return self._tokenizer.encode(text=text, add_bos=True, add_eos=False)

    def loglikelihood(
        self, requests, disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            # Truncate if sequence is too long
            if len(continuation_enc) > self.max_length:
                continuation_enc = continuation_enc[: self.max_length]
            if len(context_enc) > self.max_length:
                context_enc = context_enc[: self.max_length]

            new_reqs.append(
                (
                    (continuation, context),
                    [self._tokenizer.bos_id] + continuation_enc[::-1],
                    context_enc[1:][::-1],
                )
            )

            # new_reqs.append(
            #     ((continuation, context), continuation_enc[::-1], context_enc[::-1])
            # )

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
            """Defines the key to group and lookup one-token continuations"""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by=(
                "contexts"
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                and self.logits_cache
                else None
            ),
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )
        ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM:
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]
            count = 0
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                assert (
                    logits.shape[0] == padding_len_inp
                ), "I don't see these two would be different in any case"
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                    else None
                )

                if inplen - contlen != 0:
                    ctx_logits = logits[: (inplen - contlen)]
                    ctx_label = torch.tensor(ctx_tokens[1:], device=ctx_logits.device)
                    ctx_loss = ce_loss_fn(ctx_logits, ctx_label)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):

                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(
                        0
                    )  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()
                    cont_loss = ce_loss_fn(logits.squeeze(0), cont_toks.squeeze(0))
                    if inplen - contlen == 0:
                        full_loss = cont_loss
                        ctx_loss = cont_loss
                    else:
                        full_loss = torch.cat([cont_loss, ctx_loss])
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (
                        (
                            torch.clamp(ctx_loss, max=10).tolist(),
                            torch.clamp(cont_loss, max=10).tolist(),
                            torch.clamp(full_loss, max=10).tolist(),
                        ),
                        float(logits.sum()),
                        bool(max_equal),
                    )

                    res.append(answer)

                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)


parser = argparse.ArgumentParser(description="Main eval simulation")
parser.add_argument(
    "--model-type",
    type=str,
    choices=["multiplication", "reverse_multiplication"],
    help="multiplication, reverse_multiplication",
)
parser.add_argument("--direction", type=str, choices=["l2r", "r2l"], help="l2r/r2l")
args = parser.parse_args()


d = {
    "batch_size": 8,
    "device": "cuda",
    "dtype": "bf16",
    "seed": 1234,
    "limit": "null",
    "tasks": ["multiply"],
    "quantizer": "null",
    "checkpointer": {
        "_component_": "torchtune.training.FullModelMetaCheckpointer",
        "checkpoint_dir": f"ckpt/{args.model_type}_simulation_{args.direction}",
        "checkpoint_files": ["checkpoint.pt"],
        "model_type": "LLAMA3",
        "output_dir": "output",
    },
    "model": {"_component_": "simulation.model.llama3_2b"},
    "tokenizer": {
        "_component_": "torchtune.models.llama3.llama3_tokenizer",
        "path": "torchtitan/assets/tokenizer/original/tokenizer.model",
    },
    "reverse_eval": True,
    "reverse_template": False,
    "return_ppl": True,
    "max_seq_length": 4096,
    "desc": "new task run",
    "fullseq_prob": False,
}

task_manager = TaskManager(include_path=None)

if args.direction == "l2r":
    reverse_eval = False
else:
    reverse_eval = True

for t in range(10):
    task_dict = get_task_dict([f"multiply{t}"], task_manager)
    d["reverse_eval"] = reverse_eval
    cfg = OmegaConf.create(d)
    recipe = EleutherEvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)

    output = evaluate(
        recipe.eleuther_model_wrapper,
        task_dict,
        limit=1000,
        reverse_template=recipe._cfg.reverse_template,
        return_ppl=recipe._cfg.return_ppl,
        ppl_suffix="r2l" if recipe._cfg.reverse_eval else "l2r",
    )
