from GP_CHINA import main as GP_CHINA_MAIN
from GP_DESHARNAIS import main as GP_DESHARNAIS_MAIN
from GP_CHINA_MO import main as GP_CHINA_MO_MAIN
from GP_DESHARNAIS_MO import main as GP_DESHARNAIS_MO_MAIN

from matplotlib import pyplot as plt
import numpy as np

popSize = 500
mutation = 0.1
cx = 0.8
nGens = 100
tournSize = int(popSize / 4)

# # 20 / 80 split
print("China Dataset - Single Objective")
china_mae, china_mae_diff, china_rmse, china_rmse_diff, china_cc, china_tree = GP_CHINA_MAIN(
    popSize, mutation, cx, nGens, tournSize
)
print("Desharnais Dataset - Single Objective")
desharnais_mae, desharnais_mae_diff, desharnais_rmse, desharnais_rmse_diff, desharnais_cc, desharnais_tree = GP_DESHARNAIS_MAIN(
    popSize, mutation, cx, nGens, tournSize
)
print("China Dataset - Multi Objective")
china_mo_mae, china_mo_mae_diff, china_mo_rmse, china_mo_rmse_diff, china_mo_cc, china_mo_tree = GP_CHINA_MO_MAIN(
    popSize, mutation, cx, nGens, tournSize
)
print("Desharnais Dataset - Multi Objective")
desharnais_mo_mae, desharnais_mo_mae_diff, desharnais_mo_rmse, desharnais_mo_rmse_diff, desharnais_mo_cc, desharnais_mo_tree = GP_DESHARNAIS_MO_MAIN(
    popSize, mutation, cx, nGens, tournSize
)

desharnais_lr_mae, desharnais_lr_rmse, desharnais_lr_cc, desharnais_gaussian_mae, desharnais_gaussian_rmse, desharnais_gaussian_cc = (
    2101.555,
    2633.9681,
    0.6957,
    1911.947,
    2637.4246,
    0.6612,
)
china_lr_mae, china_lr_rmse, china_lr_cc, china_gaussian_mae, china_gaussian_rmse, china_gaussian_cc = (
    1796.1514,
    3340.4831,
    0.7273,
    3491.4531,
    4300.983,
    0.5358,
)

print(
    "Desharnais single objective - MAE : %f , RMSE = %f, Correlation Coefficient : %f"
    % (desharnais_mae, desharnais_rmse, desharnais_cc)
)
print(
    "China single objective - MAE : %f , RMSE = %f,  Correlation Coefficient : %f"
    % (china_mae, china_rmse, china_cc)
)
print(
    "Desharnais multi objective - MAE : %f , RMSE = %f,  Correlation Coefficient : %f"
    % (desharnais_mo_mae, desharnais_mo_rmse, desharnais_mo_cc)
)
print(
    "China multi objective - MAE : %f , RMSE = %f, Correlation Coefficient : %f"
    % (china_mo_mae, china_mo_rmse, china_mo_cc)
)
print(
    "Desharnais Linear Regression - MAE : %f , RMSE = %f, Correlation Coefficient : %f"
    % (desharnais_lr_mae, desharnais_lr_rmse, desharnais_lr_cc)
)
print(
    "China Linear Regression - MAE : %f , RMSE = %f, Correlation Coefficient : %f"
    % (china_lr_mae, china_lr_rmse, china_lr_cc)
)
print(
    "Desharnais Gaussian - MAE : %f , RMSE = %f, Correlation Coefficient : %f"
    % (desharnais_gaussian_mae, desharnais_gaussian_rmse, desharnais_gaussian_cc)
)
print(
    "China Linear Gaussian - MAE : %f , RMSE = %f, Correlation Coefficient : %f"
    % (china_gaussian_mae, china_gaussian_rmse, china_gaussian_cc)
)

# # china plot
# plt.figure()
# plt.xlim(0, 5000)
# plt.ylim(0, 5000)
# so = plt.scatter(china_rmse, china_mae, marker='x', color='r')
# mo = plt.scatter(china_mo_rmse, china_mo_mae, marker='s', color='g')
# lr = plt.scatter(china_lr_rmse, china_lr_mae, marker='P', color='y')
# gaus = plt.scatter(china_gaussian_rmse, china_gaussian_mae, marker='^', color='c')
# plt.suptitle("China Dataset Results")
# plt.title("Population Size = %d, Number Of Generations = %d" % (popSize, nGens))
# plt.xlabel("RMSE")
# plt.ylabel("MAE")
# plt.legend((so, mo, lr, gaus),
#                    ('Single Objective Symbolic Regression', 'Multi Objective Symbolic Regression', 'Linear Regression', 'Gaussian Processes'),
#                    scatterpoints=1,
#                    loc='upper left',
#                    ncol=1,
#                    fontsize=8)
#
# plt.figure()
# plt.xlim(0, 5000)
# plt.ylim(0, 5000)
# so = plt.scatter(desharnais_rmse, desharnais_mae, marker='x', color='r')
# mo = plt.scatter(desharnais_mo_rmse, desharnais_mo_mae, marker='s', color='g')
# lr = plt.scatter(desharnais_lr_rmse, desharnais_lr_mae, marker='P', color='y')
# gaus = plt.scatter(desharnais_gaussian_rmse, desharnais_gaussian_mae, marker='^', color='c')
# plt.suptitle("Desharnais Dataset Results")
# plt.title("Population Size = %d, Number Of Generations = %d" % (popSize, nGens))
# plt.xlabel("RMSE")
# plt.ylabel("MAE")
# plt.legend((so, mo, lr, gaus),
#                    ('Single Objective Symbolic Regression', 'Multi Objective Symbolic Regression', 'Linear Regression', 'Gaussian Processes'),
#                    scatterpoints=1,
#                    loc='upper left',
#                    ncol=1,
#                    fontsize=8)

# fake up some more data
# spread = np.random.rand(50) * 100 # difference between train and test
# center = np.ones(25) * 40 # estimated value
spread = china_mo_rmse_diff
center = china_mo_rmse
# data = np.concatenate((spread, center, flier_high, flier_low), 0)
# data = np.concatenate((spread, center), 0)
plt.figure()
plt.suptitle("CHINA Results")
plt.title("Population Size = %d, Number of Generations = %d" % (popSize, nGens))
plt.xlabel("RMSE")
plt.ylabel("MAE")
so = plt.errorbar(
    china_rmse,
    china_mae,
    xerr=[[china_rmse_diff], [china_rmse_diff]],
    yerr=[[china_mae_diff], [china_mae_diff]],
    fmt="x",
    markersize=6,
    capsize=8,
)
mo = plt.errorbar(
    china_mo_rmse,
    china_mo_mae,
    xerr=[[china_mo_rmse_diff], [china_mo_rmse_diff]],
    yerr=[[china_mo_mae_diff], [china_mo_mae_diff]],
    fmt="o",
    markersize=6,
    capsize=8,
)
lr = plt.scatter(china_lr_rmse, china_lr_mae, marker="P", color="y")
gaus = plt.scatter(china_gaussian_rmse, china_gaussian_mae, marker="^", color="c")

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(
    (so, mo, lr, gaus),
    (
        "Single Objective Symbolic Regression",
        "Multi Objective Symbolic Regression",
        "Linear Regression",
        "Gaussian Processes",
    ),
    scatterpoints=1,
    loc="upper left",
    ncol=1,
    fontsize=8,
)

# plt.errorbar(limit, GA_X, GA_E, linestyle='-', marker='^', label='GA Spread')

plt.figure()
plt.suptitle("DESHARNAIS Results")
plt.title("Population Size = %d, Number of Generations = %d" % (popSize, nGens))
plt.xlabel("RMSE")
plt.ylabel("MAE")
so = plt.errorbar(
    desharnais_rmse,
    desharnais_mae,
    xerr=[[desharnais_rmse_diff], [desharnais_rmse_diff]],
    yerr=[[desharnais_mae_diff], [desharnais_mae_diff]],
    fmt="x",
    markersize=6,
    capsize=8,
)
mo = plt.errorbar(
    desharnais_mo_rmse,
    desharnais_mo_mae,
    xerr=[[desharnais_mo_rmse_diff], [desharnais_mo_rmse_diff]],
    yerr=[[desharnais_mo_mae_diff], [desharnais_mo_mae_diff]],
    fmt="o",
    markersize=6,
    capsize=8,
)
lr = plt.scatter(desharnais_lr_rmse, desharnais_lr_mae, marker="P", color="y")
gaus = plt.scatter(
    desharnais_gaussian_rmse, desharnais_gaussian_mae, marker="^", color="c"
)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(
    (so, mo, lr, gaus),
    (
        "Single Objective Symbolic Regression",
        "Multi Objective Symbolic Regression",
        "Linear Regression",
        "Gaussian Processes",
    ),
    scatterpoints=1,
    loc="upper left",
    ncol=1,
    fontsize=8,
)

# plt.errorbar(limit, GA_X, GA_E, linestyle='-', marker='^', label='GA Spread')

plt.show()
#
# Linear Regression Model Desharnais
#
# Effort =
#
#    -456.3686 * TeamExp +
#     368.8159 * ManagerExp +
#     172.0017 * Length +
#      10.2841 * Transactions +
#      10.9585 * Entities +
#     107.6351 * Adjustment +
#    4383.0672 * Language=2,1 +
#   -6925.9739
#
# Time taken to build model: 0 seconds
#
# === Evaluation on test split ===
#
# Time taken to test model on test split: 0.01 seconds
#
# === Summary ===
#
# Correlation coefficient                  0.6957
# Mean absolute error                   2101.555
# Root mean squared error               2633.9681
# Relative absolute error                 80.7206 %
# Root relative squared error             76.786  %
# Total Number of Instances               16
#


# Gaussian Prcesses Desharnais
#
# === Evaluation on test split ===
#
# Time taken to test model on test split: 0.02 seconds
#
# === Summary ===
#
# Correlation coefficient                  0.6612
# Mean absolute error                   1911.9477
# Root mean squared error               2637.5246
# Relative absolute error                 73.4378 %
# Root relative squared error             76.8897 %
# Total Number of Instances               16

#
# Linear Regression Model China
#
# Effort =
#
#       6.1591 * AFP +
#      -3.596  * Input +
#       8.6237 * Enquiry +
#      -5.7689 * File +
#     483.7636 * PDR_AFP +
#    -294.1781 * PDR_UFP +
#    -552.7505 * NPDR_AFP +
#     527.1438 * NPDU_UFP +
#     661.7625 * Resource +
#     139.6964 * Duration +
#   -2653.7709
#
# Time taken to build model: 0 seconds
#
# === Evaluation on test split ===
#
# Time taken to test model on test split: 0 seconds
#
# === Summary ===
#
# Correlation coefficient                  0.7273
# Mean absolute error                   1796.1514
# Root mean squared error               3340.4831
# Relative absolute error                 57.4387 %
# Root relative squared error             77.189  %
# Total Number of Instances              100


# Gaussian processes china
#
# === Evaluation on test split ===
#
# Time taken to test model on test split: 0.08 seconds
#
# === Summary ===
#
# Correlation coefficient                  0.5358
# Mean absolute error                   3491.4531
# Root mean squared error               4300.983
# Relative absolute error                111.6524 %
# Root relative squared error             99.3834 %
# Total Number of Instances              100
#
