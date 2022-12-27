import pickle
import numpy as np

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
    for one_time_string in time_string_list:
        info_path = "saves/train/{0}_{1}/{0}_{1}_info.npy".format(model_name, one_time_string)
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        print("{}\t{}".format(
            sum(info["loss"][-average_length:]) / average_length,
            sum(info["real_loss"][-average_length:]) / average_length,
        ))
    print()

if __name__ == "__main__":
    sir_origin_time_strings = ["20221226_001416","20221226_002338","20221226_003325","20221226_004312","20221226_005253","20221226_010229","20221226_011214","20221226_012134","20221226_013120","20221226_014122"]
    sir_plan3_time_strings = ["20221226_001359","20221226_002805","20221226_004222","20221226_005641","20221226_011053","20221226_012509","20221226_013916","20221226_015339","20221226_020742","20221226_022141"]
    one_time_detail_check("SIR_Fourier_Lambda", sir_origin_time_strings, 10000)
    one_time_detail_check("SIR_Fourier_Lambda", sir_plan3_time_strings, 10000)
    # one_time_build_from_record()
    pass
