from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn
from trl import DPOTrainer as TRLDPOTrainer


class DPOTrainer(TRLDPOTrainer):
    def __init__(self, *args, refine_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.refine_token_id = refine_token_id

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.padding_value
        )

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch[
                "pixel_attention_mask"
            ]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask), dim=1
            )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = (
                torch.nonzero(empty_cols)[0].item()
                if empty_cols.any()
                else attention_mask.size(1) + 1
            )
            input_ids = input_ids[:, : first_empty_col - 1]
            attention_mask = attention_mask[:, : first_empty_col - 1]
            loss_mask = loss_mask[:, : first_empty_col - 1]

            # Truncate right
            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, **model_kwargs
            )

            # Offset the logits by one to align with the labels
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = (
            0  # dummy token; we'll ignore the losses on these tokens later
        )
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(
                    2 * logprobs, dim=-1
                )  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(
                    -1
                ) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(
                    torch.exp(chosen_weights + rejected_weights), max=1
                )

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            # ! Edited
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            has_refine_token = self.refine_token_id is not None

            if has_refine_token:
                # ? This code would not work in the multi-turn setting

                # Check for occurrences of refine token
                marker = torch.flip(
                    (chosen_labels == self.refine_token_id), dims=[1]
                ).cumsum(dim=1)
                marker = torch.flip(marker, dims=[1])  # Flip back to original order
                marker = torch.roll(marker, shifts=-1, dims=1)  # Shift left
                marker[:, -1] = 0  # Set the last element to 0
                mask = marker > 0  # Elements before refine token

                # Mask rows without refine token
                row_has_refine = (
                    (chosen_labels == self.refine_token_id).any(dim=1).unsqueeze(1)
                )  # Identify rows containing refine token
                mask = (
                    mask & row_has_refine
                )  # Only apply mask to rows with refine token

                # Apply -100 to elements before refine token, leaving other rows untouched
                chosen_labels = torch.where(
                    mask, torch.full_like(chosen_labels, 0), chosen_labels
                )

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][
            loss_mask[:num_examples]
        ].mean()
        output["mean_rejected_logits"] = logits[num_examples:][
            loss_mask[num_examples:]
        ].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output
