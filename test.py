import pickle
import numpy as np
import torch

def one_unit(string):
    skip_words = ["init"]
    if "=" not in string:
        return True, None, string
    parts = string.split("=")
    if parts[0] in skip_words:
        return False, parts[0], parts[1]
    else:
        return True, parts[0], parts[1]


def one_time_build_from_record():
    with open("test/record.txt", "r") as f:
        lines = f.readlines()
    with open("test/record.csv", "w") as f:
        for one_line in lines:
            parts = one_line.split()
            for one_part in parts:
                flag, part_name, part_val = one_unit(one_part)
                if flag:
                    f.write("{},".format(part_val))
            f.write("\n")


def one_time_detail_check(model_name, time_string_list, average_length):
    print("loss:")
    for one_time_string in time_string_list:
        info_path = "saves/train/{0}_{1}/{0}_{1}_info.npy".format(model_name, one_time_string)
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        print("{}".format(
            sum(info["loss"][-average_length:]) / average_length,
        ))
    print("real loss:")
    for one_time_string in time_string_list:
        info_path = "saves/train/{0}_{1}/{0}_{1}_info.npy".format(model_name, one_time_string)
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        print("{}".format(
            sum(info["real_loss"][-average_length:]) / average_length,
        ))
    print()

if __name__ == "__main__":
    # sir_origin_time_strings = ["20221226_001416","20221226_002338","20221226_003325","20221226_004312","20221226_005253","20221226_010229","20221226_011214","20221226_012134","20221226_013120","20221226_014122"]
    # sir_plan3_time_strings = ["20221226_001359","20221226_002805","20221226_004222","20221226_005641","20221226_011053","20221226_012509","20221226_013916","20221226_015339","20221226_020742","20221226_022141"]
    # rep_original_time_strings = ["20221226_002700","20221226_014331","20221226_030112","20221226_042011","20221226_053758","20221226_065546","20221226_081128","20221226_092939","20221226_104616","20221226_120047"]
    # rep_plan3_time_strings = ["20221226_002656","20221226_021123","20221226_035405","20221226_053835","20221226_072055","20221226_090353","20221226_104703","20221226_123144","20221226_141414","20221226_155656"]
    # cc1_original_time_strings = ["20221226_002758","20221226_002937","20221226_003116","20221226_003254","20221226_003432","20221226_003610","20221226_003748","20221226_003927","20221226_004106","20221226_004244","20221226_004422","20221226_004600","20221226_004738","20221226_004915","20221226_005054","20221226_005233","20221226_005410","20221226_005549","20221226_005727","20221226_005906"]
    # cc1_plan3_time_strings = ["20221226_002748","20221226_003021","20221226_003256","20221226_003538","20221226_003817","20221226_004054","20221226_004328","20221226_004554","20221226_004837","20221226_005103","20221226_005332","20221226_005557","20221226_005841","20221226_010124","20221226_010350","20221226_010617","20221226_010846","20221226_011112","20221226_011341","20221226_011613"]
    # print("original:")
    # one_time_detail_check("CC1_Fourier_Lambda", cc1_original_time_strings, 1000)
    # print("plan3:")
    # one_time_detail_check("CC1_Fourier_Lambda", cc1_plan3_time_strings, 1000)
    criterion = torch.nn.MSELoss(reduce=False)
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([3.0, 4.0, 5.0])
    print(criterion(a, b))


    # parts = string1.split()
    # print("\",\"".join(parts))
    # print("original:")
    # one_time_detail_check("REP_Fourier_Lambda", rep_original_time_strings, 10000)
    # print("plan3:")
    # one_time_detail_check("REP_Fourier_Lambda", rep_plan3_time_strings, 10000)
    # print("original:")
    # one_time_detail_check("SIR_Fourier_Lambda", sir_origin_time_strings, 10000)
    # print("plan3:")
    # one_time_detail_check("SIR_Fourier_Lambda", sir_plan3_time_strings, 10000)
    # one_time_build_from_record()
    pass
