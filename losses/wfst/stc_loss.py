import math
from typing import List, Literal
import k2
import torch

from .utils import retain_and_map_tokens_in_batch, encode_text_supervisions


class STCGraphCompiler:
    def compile(
        self,
        batch: List[torch.Tensor],
        token_penalty: float,
        device: torch.device | str,
    ) -> k2.Fsa:
        assert len(batch) >= 2
        targets, lengths = batch[:2]
        max_token = -1

        for target, length in zip(targets, lengths):
            # Truncate target to actual length
            target = target[:length]
            max_token = max(int(target.max().item()), max_token)

        # Convert to lists for processing
        target_lists = []
        for target, length in zip(targets, lengths):
            target_lists.append(target[:length].tolist())

        star_idx = max_token + 1
        graphs = [
            self.build_single_stc_graph(target, star_idx, token_penalty, device)
            for target in target_lists
        ]
        graphs = k2.Fsa.from_fsas(graphs)

        assert graphs.requires_grad is False
        return graphs

    def build_single_stc_graph(
        self, target: List[int], star_idx: int, prob: float, device: torch.device | str
    ):
        """Build STC Label Graph for a single target sequence following GTN approach."""
        STC_BLANK_IDX = 0
        L = len(target)
        S = 2 * L + 1

        arcs = []

        # Create self-less CTC graph structure
        for l in range(S):
            idx = (l - 1) // 2
            label = target[idx] if l % 2 else STC_BLANK_IDX

            # Self-loop for blank tokens
            if label == STC_BLANK_IDX:
                arcs.append(self.build_arc(l, l, label, 0.0))

            # Forward arc
            if l > 0:
                arcs.append(self.build_arc(l - 1, l, label, 0.0))

            # Skip arc for non-blank tokens
            if l % 2 and l > 1:
                arcs.append(self.build_arc(l - 2, l, label, 0.0))

        # Add extra nodes and arcs for STC
        node_offset = S  # Start numbering new nodes after CTC nodes
        log_prob = math.log(prob)

        for l in range(L + 1):
            p1 = 2 * l - 1
            p2 = 2 * l
            c1 = node_offset + l  # New node for this position

            # Determine the star token index
            idx = star_idx if l == L else (star_idx + target[l])

            # Arcs to the new node
            if p1 >= 0:
                arcs.append(self.build_arc(p1, c1, idx, log_prob))
            arcs.append(self.build_arc(p2, c1, idx, log_prob))

            # Self-loop on the new node
            arcs.append(self.build_arc(c1, c1, idx, log_prob))

            # Arc back to CTC path
            if l < L:
                arcs.append(self.build_arc(c1, 2 * l + 1, target[l], 0.0))
            arcs.append(self.build_arc(c1, p2, STC_BLANK_IDX, 0.0))

        # Add final states
        final_state = node_offset + L + 1
        arcs.append(
            self.build_arc(S - 1, final_state, -1, 0.0)
        )  # Final state from CTC path
        arcs.append(
            self.build_arc(S - 2, final_state, -1, 0.0)
        )  # Final state from CTC path
        arcs.append(
            self.build_arc(node_offset + L, final_state, -1, 0.0)
        )  # Final state from last STC node
        arcs.append(f"{final_state}")  # Mark as final state

        arcs.sort(key=lambda x: int(x.split()[0]))

        # Combine arcs and state definitions
        fsa_string = "\n".join(arcs)

        fsa = k2.Fsa.from_str(fsa_string).to(device)
        fsa = k2.arc_sort(fsa)
        return fsa

    def build_arc(
        self, start_state: int, end_state: int, output_idx: int, score: float = 0.0
    ) -> str:
        return f"{start_state} {end_state} {output_idx} {score}"


class STCLoss(torch.nn.Module):
    def __init__(
        self,
        output_beam: float = 10.0,
        use_double_scores: bool = True,
        reduction: Literal["none", "mean", "sum"] = "mean",
        p0: float = 0.5,
        plast: float = 0.9,
        thalf: float = 10000,
        init_scale=5.0,
        clamp_min=1.0,
        clamp_max=10.0,
    ) -> None:
        super().__init__()
        self.compiler = STCGraphCompiler()
        self.output_beam = output_beam
        self.reduction = reduction
        self.use_double_scores = use_double_scores

        self.p0 = p0
        self.plast = plast
        self.thalf = thalf
        self.nstep = 0

        self.raw_scale = torch.nn.Parameter(torch.tensor(float(init_scale)))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @staticmethod
    def logsubexp(a, b):
        """
        Compute log(exp(a) - exp(b)) in a numerically stable way.

        This is used for computing star token probabilities in STC.
        For STC, we want log(P(star) - P(specific_token)) where:
        - P(star) = sum of all non-blank token probabilities
        - P(specific_token) = probability of a specific token

        Args:
            a: log probabilities of star token [B, T, 1]
            b: log probabilities of specific tokens [B, T, num_tokens]

        Returns:
            log(exp(a) - exp(b)) computed stably
        """
        # Ensure a and b are broadcastable
        # a should be [B, T, 1], b should be [B, T, num_tokens]

        # For numerical stability, we use the identity:
        # log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
        # But we need to be careful when b > a (which would make exp(b-a) > 1)

        # Method 1: Use the stable formulation
        # When a >= b: log(exp(a) - exp(b)) = a + log(1 - exp(b - a))
        # When a < b: result is undefined (negative inside log)

        diff = b - a  # [B, T, num_tokens]

        # Clamp to avoid exp(diff) > 1 which would cause log(negative)
        # If diff > 0, then exp(a) < exp(b), so exp(a) - exp(b) < 0
        # In this case, we set a small negative value
        diff = torch.clamp(diff, max=0.0)  # Ensure diff <= 0

        # Now compute log(1 - exp(diff)) safely
        # Use log1p(-exp(diff)) = log(1 - exp(diff)) for better numerical stability
        result = a + torch.log1p(-torch.exp(diff))

        return result

    def forward(self, logits: torch.Tensor, batch: List[torch.Tensor]):
        """
        Compute STC loss.

        Args:
            logits: Tensor of shape (B, T, V) where B is batch size,
                   T is sequence length, V is vocabulary size
            batch: List containing [targets, lengths] where targets is (B, L)
                  and lengths is (B,)

        Returns:
            STC loss tensor
        """
        B, T, C = logits.shape
        targets, lengths = batch[1], batch[2]
        device = logits.device

        if self.training:
            self.nstep += 1
        prob = self.plast + (self.p0 - self.plast) * math.exp(
            -self.nstep * math.log(2) / self.thalf
        )

        with torch.set_grad_enabled(logits.requires_grad):
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Remap targets to reduce vocabulary size
            # Select only the tokens present in current batch plus blank and star tokens
            STC_BLANK_IDX = 0

            targets, select_idx_tensor = retain_and_map_tokens_in_batch(
                targets, lengths, STC_BLANK_IDX, log_probs.device
            )

            # Build STC decoding graphs with remapped targets
            decoding_graphs = self.compiler.compile([targets, lengths], prob, device)
            
            supervision_segments = encode_text_supervisions(
                targets, T, return_indices=False
            )

            scale = torch.clamp(self.raw_scale, self.clamp_min, self.clamp_max)

            # <star> token
            lse = torch.logsumexp(scale * log_probs[:, :, 1:], 2, keepdim=True)

            # Reduce logits to only selected tokens
            log_probs = log_probs.index_select(2, select_idx_tensor)

            # <star>\tokens for all tokens present in current batch
            neglse = STCLoss.logsubexp(lse, scale * log_probs[:, :, 1:])

            log_probs = torch.cat([log_probs, lse, neglse], dim=2)

            # Create dense FSA vector from reduced logits
            dense_fsa_vec = k2.DenseFsaVec(
                log_probs, supervision_segments, allow_truncate=0
            )

            # Pass target_lengths when reduction is 'mean'
            target_lengths = lengths.to(device) if self.reduction == "mean" else None

        # Compute STC loss using k2's CTC loss (STC graphs are compatible)
        stc_loss = k2.ctc_loss(
            decoding_graph=decoding_graphs,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.output_beam,
            reduction=self.reduction,
            use_double_scores=self.use_double_scores,
            target_lengths=target_lengths,
        )

        assert stc_loss.requires_grad == logits.requires_grad

        return {"loss": stc_loss}
