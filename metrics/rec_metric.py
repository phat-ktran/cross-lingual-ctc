import math
from rapidfuzz.distance import Levenshtein
import string


class RecMetric(object):
    def __init__(
        self, main_indicator="acc", is_filter=False, ignore_space=True, **kwargs
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
        return text.lower()

    def _correct_rate(self, pred: str, target: str) -> float:
        Nt = len(target)
        if Nt == 0:
            return 1.0 if len(pred) == 0 else 0.0
        if len(pred) == 0:
            return 0.0
        ds_es = Levenshtein.distance(pred, target, weights=(0, 1, 1))
        return (Nt - ds_es) / Nt

    def _accurate_rate(self, pred: str, target: str) -> float:
        Nt = len(target)
        if Nt == 0:
            return 1.0 if len(pred) == 0 else float("-inf")
        total_errors = Levenshtein.distance(pred, target, weights=(1, 1, 1))
        return (Nt - total_errors) / Nt

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num, correct_char_num = 0, 0
        all_num, all_char_num = 0, 0
        norm_edit_dis = 0.0
        corr_rate_sum, acc_rate_sum = 0.0, 0.0
        acc_sq_sum, acc_rate_sq_sum = 0.0, 0.0

        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)

            # per-sample metrics
            acc_i = 1.0 if pred == target else 0.0
            ar_i = self._accurate_rate(pred, target)
            cr_i = self._correct_rate(pred, target)
            nd_i = Levenshtein.normalized_distance(pred, target)

            # accumulate sample-level
            acc_sq_sum += acc_i * acc_i
            acc_rate_sq_sum += ar_i * ar_i

            corr_rate_sum += cr_i
            acc_rate_sum += ar_i
            norm_edit_dis += nd_i

            if acc_i == 1.0:
                correct_num += 1

            max_len = max(len(target), len(pred))
            for i in range(max_len):
                pred_c = pred[i] if len(pred) > i else None
                target_c = target[i] if len(target) > i else None
                if pred_c == target_c:
                    correct_char_num += 1
            all_char_num += len(target)
            all_num += 1

        # update global sums
        self.correct_num += correct_num
        self.correct_char_num += correct_char_num
        self.all_num += all_num
        self.all_char_num += all_char_num
        self.norm_edit_dis += norm_edit_dis
        self.corr_rate += corr_rate_sum
        self.acc_rate += acc_rate_sum
        self.acc_sq += acc_sq_sum
        self.acc_rate_sq += acc_rate_sq_sum

        return {
            "acc": correct_num / (all_num + self.eps),
            "acc_rate": acc_rate_sum / (all_num + self.eps),
            "char_acc": correct_char_num / (all_char_num + self.eps),
            "norm_edit_dis": 1 - norm_edit_dis / (all_num + self.eps),
            "corr_rate": corr_rate_sum / (all_num + self.eps),
        }

    def get_metric(self):
        # compute means
        N = self.all_num + self.eps
        mean_acc = self.correct_num / N
        mean_acc_rate = self.acc_rate / N

        # compute stds
        E_x2 = self.acc_sq / N
        var_acc = max(E_x2 - mean_acc * mean_acc, 0.0)
        std_acc = math.sqrt(var_acc)

        E_ar2 = self.acc_rate_sq / N
        var_ar = max(E_ar2 - mean_acc_rate * mean_acc_rate, 0.0)
        std_acc_rate = math.sqrt(var_ar)

        # compute other metrics
        char_acc = 1.0 * self.correct_char_num / (self.all_char_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / N
        corr_rate = self.corr_rate / N
        acc_rate = mean_acc_rate

        self.reset()
        return {
            "acc": mean_acc,
            "acc_std": std_acc,
            "acc_rate": acc_rate,
            "acc_rate_std": std_acc_rate,
            "char_acc": char_acc,
            "norm_edit_dis": norm_edit_dis,
            "corr_rate": corr_rate,
        }

    def reset(self):
        self.correct_num = 0
        self.correct_char_num = 0
        self.all_num = 0
        self.all_char_num = 0
        self.norm_edit_dis = 0
        self.corr_rate = 0
        self.acc_rate = 0
        self.acc_sq = 0
        self.acc_rate_sq = 0
