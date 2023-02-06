import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, help="start")
    parser.add_argument("--end", type=int, help="end")
    opt = parser.parse_args()
    for job_id in range(opt.start, opt.end + 1):
        os.system("scancel {0:d}".format(job_id))
