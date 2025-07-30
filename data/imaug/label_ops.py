import copy
import json
import numpy as np

from losses.softctc.models.connections import Connections


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        lower=False,
    ):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super(CTCLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data["label_ace"] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class SoftCTCLabelEncode(BaseRecLabelEncode):
    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super().__init__(max_text_length, character_dict_path, use_space_char)

    def encode(self, text):
        # Input validation
        if not text or len(text) == 0:
            return None
        
        # Parse JSON
        try:
            confusion_network = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        
        # Validate structure
        if not isinstance(confusion_network, list):
            return None
        
        # Check length limit
        if len(confusion_network) > self.max_text_len:
            return None
        
        # Apply lowercase if needed
        if self.lower:
            for i in range(len(confusion_network)):
                if isinstance(confusion_network[i], dict):
                    confusion_network[i] = {
                        (k.lower() if isinstance(k, str) else k): v 
                        for k, v in confusion_network[i].items()
                    }
        
        # Convert characters to dictionary indices
        for i in range(len(confusion_network)):
            if not isinstance(confusion_network[i], dict):
                continue
                
            new_confusion_set = {}
            has_valid_chars = False
            
            for char, prob in confusion_network[i].items():
                if char in self.dict:
                    new_confusion_set[self.dict[char]] = prob
                    has_valid_chars = True
                # If char not in dict, we skip it (unknown character)
            
            # If no valid characters found, add None token
            if not has_valid_chars:
                new_confusion_set[None] = 1.0
            
            confusion_network[i] = new_confusion_set
        
        # Check if all positions are None (completely unknown text)
        if all(len(obj) == 1 and None in obj for obj in confusion_network):
            return None
        
        return confusion_network

    def __call__(self, data):
        text = data["label"]
        confusion_network = self.encode(text)
        if confusion_network is None:
            return None
        text = []
        for confusion_set in confusion_network:
            for char, _ in confusion_set.items():
                if char is None:
                    continue
                text.append(char)
        connections = Connections.from_confusion_network(confusion_network, 0)
        data["label"] = np.array(text)
        data["connections"] = connections
        data["length"] = np.array(len(confusion_network))
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character

class MultiLabelEncode(BaseRecLabelEncode):
    def __init__(
        self,
        max_text_length,
        character_dict_path=None,
        use_space_char=False,
        gtc_encode=None,
        **kwargs,
    ):
        super(MultiLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

        self.ctc_encode = CTCLabelEncode(
            max_text_length, character_dict_path, use_space_char, **kwargs
        )
        self.gtc_encode_type = gtc_encode

        self.gtc_encode = eval(gtc_encode)(
            max_text_length, character_dict_path, use_space_char, **kwargs
        )

    def __call__(self, data):
        data_ctc = copy.deepcopy(data)
        data_gtc = copy.deepcopy(data)
        data_out = dict()
        data_out["img_path"] = data.get("img_path", None)
        data_out["image"] = data["image"]
        ctc = self.ctc_encode.__call__(data_ctc)
        gtc = self.gtc_encode.__call__(data_gtc)
        if ctc is None or gtc is None:
            return None
        data_out["label_ctc"] = ctc["label"]
        if self.gtc_encode_type is not None:
            data_out["label_gtc"] = gtc["label"]
        else:
            data_out["label_sar"] = gtc["label"]
        data_out["length"] = ctc["length"]
        return data_out


class NRTRLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs
    ):
        super(NRTRLabelEncode, self).__init__(
            max_text_length, character_dict_path, use_space_char
        )

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["length"] = np.array(len(text))
        text.insert(0, 2)
        text.append(3)
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank", "<unk>", "<s>", "</s>"] + dict_character
        return dict_character
