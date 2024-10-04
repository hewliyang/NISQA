# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin

This script evaluates a trained speech quality model on a given Dataset.

If a 'csv_con' CSV file with per-condition results is provided, both the
per-file CSV and per-condition CSV need a column with the name 'con' that
contains the condition number. If 'csv_con' stays empty, only per-file results
are calculated.

"""
from nisqa.NISQA_model import nisqaModel

if __name__ == "__main__":

    args = {
        "mode": "predict_csv",
        "pretrained_model": "artifacts/transfer-learning_240425_214232672655/transfer-learning_240425_214232672655.tar",
        "data_dir": "train_data/",
        "output_dir": "eval/",
        "csv_file": "TRAIN.csv",
        "csv_deg": "filepath_deg",
        "csv_mos_val": "mos",
        "tr_num_workers": 6,
        "tr_bs_val": 40,
        "ms_channel": None,
    }

    nisqa = nisqaModel(args)
    nisqa.predict()
    nisqa.evaluate(mapping="first_order", do_print=True, do_plot=True)
