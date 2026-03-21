#!/usr/bin/env python3
"""
Patch PaddleOCR quant.py and export_model.py to support NRTRLoss.

PP-OCRv5 uses CTC+NRTR dual head, but the quantization scripts hardcode
an assertion that the second loss must be SARLoss. This patch makes them
handle both SARLoss and NRTRLoss.
"""
import sys
import re

OLD_BLOCK = '''        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            # update SARLoss params
            assert list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss"
            if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                    "ignore_index": char_num + 1
                }
            else:
                config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                    char_num + 1
                )
            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list'''

NEW_BLOCK = '''        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            second_loss_name = list(config["Loss"]["loss_config_list"][1].keys())[0]
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if second_loss_name == "SARLoss":
                # update SARLoss params
                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {
                        "ignore_index": char_num + 1
                    }
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (
                        char_num + 1
                    )
                out_channels_list = {}
                out_channels_list["CTCLabelDecode"] = char_num
                out_channels_list["SARLabelDecode"] = char_num + 2
                config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
            elif second_loss_name == "NRTRLoss":
                # PP-OCRv5 uses CTC+NRTR dual head
                out_channels_list = {}
                out_channels_list["CTCLabelDecode"] = char_num
                out_channels_list["NRTRLabelDecode"] = char_num + 2
                config["Architecture"]["Head"]["out_channels_list"] = out_channels_list'''

for filepath in sys.argv[1:]:
    with open(filepath, 'r') as f:
        content = f.read()

    if OLD_BLOCK in content:
        content = content.replace(OLD_BLOCK, NEW_BLOCK)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Patched: {filepath}")
    else:
        print(f"Skipped (pattern not found): {filepath}")
