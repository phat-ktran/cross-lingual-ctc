from typing import List, Literal
import k2
import torch


from .utils import encode_text_supervisions_with_length, retain_and_map_tokens_in_batch, encode_text_supervisions


class CTCGraphCompiler:    
    def compile(self, batch: List[torch.Tensor], device: torch.device | str) -> k2.Fsa:
        assert len(batch) >= 2
        targets, lengths = batch[:2]
        targets, lengths = targets.tolist(), lengths.tolist()

        for i, length in enumerate(lengths):
            targets[i] = targets[i][:length]

        decoding_graph = k2.ctc_graph(
            [targets] if targets is List[int] else targets,
            modified=False,
            device=device,
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph


class WfstCTCLoss(torch.nn.Module):
    def __init__(
        self,
        output_beam: float = 5.0,
        use_double_scores: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self.compiler = CTCGraphCompiler()
        self.output_beam = output_beam
        self.reduction = reduction
        self.use_double_scores = use_double_scores

    def forward(self, logits: torch.Tensor, batch: List[torch.Tensor]):
        targets, lengths = batch[1], batch[2]
        B, T, V = logits.shape

        with torch.set_grad_enabled(logits.requires_grad):
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Remap targets to reduce vocabulary size
            # Select only the tokens present in current batch plus blank and star tokens
            CTC_BLANK_IDX = 0

            targets, select_idx_tensor = retain_and_map_tokens_in_batch(
                targets, lengths, CTC_BLANK_IDX, log_probs.device
            )

            # Reduce logits to only selected tokens
            log_probs = log_probs.index_select(2, select_idx_tensor)

            decoding_graphs = self.compiler.compile(
                [targets, lengths], device=log_probs.device
            )

            supervision_segments = encode_text_supervisions(
                targets, T, return_indices=False
            )

            dense_fsa_vec = k2.DenseFsaVec(
                log_probs, supervision_segments, allow_truncate=0
            )

            # Pass target_lengths when reduction is 'mean'
            target_lengths = lengths.to(log_probs.device) if self.reduction == "mean" else None

            loss = k2.ctc_loss(
                decoding_graph=decoding_graphs,
                dense_fsa_vec=dense_fsa_vec,
                output_beam=self.output_beam,
                reduction=self.reduction,
                use_double_scores=self.use_double_scores,
                target_lengths=target_lengths,
            )
            
            if self.reduction == "none":
                loss = torch.sum(
                    torch.where(
                        loss != float("inf"),
                        loss,
                        torch.tensor(0, dtype=torch.float32).to(log_probs.device),
                    )
                )

        assert loss.requires_grad == logits.requires_grad

        return {"loss": loss}
