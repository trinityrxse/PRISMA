import glob
from analysis import *

root_signal_SM = glob.glob(
    "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/SM/WWW01j_000[0-4].root")
root_bkg = glob.glob(
    "/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/3l01j/3l01j_00[0-1]?.root")
name_SM = ""

root_signal_cHq1_m1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cHq1_-1/WWW01j_000[0-4].root")
name_cHq1_m1 = "cHq1_-1"

root_signal_cHq1_p1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cHq1_1/WWW01j_000[0-4].root")
name_cHq1_p1 = "cHq1_1"

root_signal_cHq3_m1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cHq3_-1/WWW01j_000[0-4].root")
name_cHq3_m1 = "cHq3_-1"

root_signal_cHq3_p1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cHq3_1/WWW01j_000[0-9].root")
name_cHq3_p1 = "cHq3_1"

root_signal_cHW_m1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cHW_-1/WWW01j_00[0-1]?.root")
name_cHW_m1 = "cHW_-1"

root_signal_cHW_p1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cHW_1/WWW01j_00[0-1]?.root")
name_cHW_p1 = "cHW_1"

root_signal_cW_m1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cW_-1/WWW01j_00[0-1]?.root")
name_cW_m1 = "cW_-1"

root_signal_cW_p1 = glob.glob(
    r"/Users/trinitystenhouse/Documents/University_MSci/2022-3/Samples/Signal/sig_cW_1/WWW01j_00[0-1]?.root")
name_cW_p1 = "cW_1"

data_list = [[root_signal_cHq1_m1, name_cHq1_m1], [root_signal_cHq1_p1, name_cHq1_p1], [root_signal_cHq3_m1, name_cHq3_m1],
             [root_signal_cHq3_p1, name_cHq3_p1], [root_signal_cHW_m1, name_cHW_m1], [root_signal_cHW_p1, name_cHW_p1],
             [root_signal_cW_m1, name_cW_m1], [root_signal_cW_p1, name_cW_p1]]

for i in range(0, len(data_list)):
    pre_processing(data_list[i][0], root_bkg, data_list[i][1])



