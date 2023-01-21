#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import itertools
import autograd.numpy as npg
from bokeh.io import show
from bokeh.models import ColumnDataSource, CrosshairTool, HoverTool, DataTable
from bokeh.models import DatetimeTickFormatter, NumeralTickFormatter, Span, TableColumn
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.transform import cumsum
from bokeh.colors import RGB
import glob
import os
import sys
import time
from datetime import datetime
import math
import requests
import json
from typing import List, Tuple, Set
import yfinance as yf
import colorsys

from scipy.optimize import minimize

stocks_read = pd.read_csv("configs/stocks.csv", "\t", index_col=0, header=None).loc[
    :, :2
]
stocks_limit = 100
stocks = {}
stocks["ticks"] = list(stocks_read.loc[:, 2])[:stocks_limit]
stocks["names"] = list(stocks_read.loc[:, 1])[:stocks_limit]

CASH_OVERIDE = 8058.68

funds = {
    "ticks": (
        [
            list(l_)
            for l_ in zip(
                *[
                    #     ('FFEBX', 'Fidelity Environmental Bond'),
                    #     ('FLOWX', 'Fidelity Water Sustainability'),
                    #     ('FSLEX', 'Fidelity Environment & Alternative Energy Fund'),
                    #     ('FCAEX', 'Fidelity Climate Action Fund'),
                    #     ('VFTAX', 'Vanguard FTSE Social Index Fund'),
                    ("FXAIX", "Fidelity 500 Index Fund"),
                    #     ('FFSFX', 'Fidelity Freedom 2065 Fund'),
                    #     ('ACLTX', 'American Century NT Growth Fund G Class'),
                    #     ('TRZBX', 'T. Rowe Price Blue Chip Growth Fund Z Class'),
                    #     ('TILWX', 'TIAA-CREF Large Cap Growth Fund Class W'),
                    ("FITLX", "Fidelity US Sustainability Index Fund"),
                    ("FNIDX", "Fidelity International Sustainability Index Fd"),
                    ("FNDSX", "Fidelity Sustainability Bond Index Fund"),
                    ("FSEBX", "Fidelity Sustainable U.S. Equity Fund"),
                    #     ('FWOMX', "Fidelity Women's Leadership")
                ]
            )
        ]
    )[0],
    "names": (
        [
            list(l_)
            for l_ in zip(
                *[
                    #     ('FFEBX', 'Fidelity Environmental Bond'),
                    #     ('FLOWX', 'Fidelity Water Sustainability'),
                    #     ('FSLEX', 'Fidelity Environment & Alternative Energy Fund'),
                    #     ('FCAEX', 'Fidelity Climate Action Fund'),
                    #     ('VFTAX', 'Vanguard FTSE Social Index Fund'),
                    ("FXAIX", "Fidelity 500 Index Fund"),
                    #     ('FFSFX', 'Fidelity Freedom 2065 Fund'),
                    #     ('ACLTX', 'American Century NT Growth Fund G Class'),
                    #     ('TRZBX', 'T. Rowe Price Blue Chip Growth Fund Z Class'),
                    #     ('TILWX', 'TIAA-CREF Large Cap Growth Fund Class W'),
                    ("FITLX", "Fidelity US Sustainability Index Fund"),
                    ("FNIDX", "Fidelity International Sustainability Index Fd"),
                    ("FNDSX", "Fidelity Sustainability Bond Index Fund"),
                    ("FSEBX", "Fidelity Sustainable U.S. Equity Fund"),
                    #     ('FWOMX', "Fidelity Women's Leadership")
                ]
            )
        ]
    )[1],
}

# In[10]:


etfs = {
    "ticks": (
        [
            list(l_)
            for l_ in zip(
                *[
                    ("XLRE", "The Real Estate Select Sector SPDR Fund"),
                    ("XLV", "Health Care Select Sector SPDR Fund"),
                    (
                        "FLSW",
                        "Franklin Templeton ETF Trust - Franklin FTSE Switzerland ETF",
                    ),
                    ("FCOM", "Fidelity MSCI Communication Services Index ETF"),
                    ("SUSA", "iShares Trust - iShares MSCI USA ESG Select ETF"),
                    ("IQSU", "IQ Candriam ESG US Equity ETF"),
                    ("USSG", "Xtrackers MSCI USA ESG Leaders Equity ETF"),
                    (
                        "SUSB",
                        "iShares Trust - iShares ESG Aware 1-5 Year USD Corporate Bond ETF",
                    ),
                    ("SNPE", "Xtrackers S&P 500 ESG ETF"),
                    ("SUSL", "iShares Trust - iShares ESG MSCI USA Leaders ETF"),
                    (
                        "EAGG",
                        "iShares Trust - iShares ESG Aware U.S. Aggregate Bond ETF",
                    ),
                ]
            )
        ]
    )[0],
    "names": (
        [
            list(l_)
            for l_ in zip(
                *[
                    ("XLRE", "The Real Estate Select Sector SPDR Fund"),
                    ("XLV", "Health Care Select Sector SPDR Fund"),
                    (
                        "FLSW",
                        "Franklin Templeton ETF Trust - Franklin FTSE Switzerland ETF",
                    ),
                    ("FCOM", "Fidelity MSCI Communication Services Index ETF"),
                    ("SUSA", "iShares Trust - iShares MSCI USA ESG Select ETF"),
                    ("IQSU", "IQ Candriam ESG US Equity ETF"),
                    ("USSG", "Xtrackers MSCI USA ESG Leaders Equity ETF"),
                    (
                        "SUSB",
                        "iShares Trust - iShares ESG Aware 1-5 Year USD Corporate Bond ETF",
                    ),
                    ("SNPE", "Xtrackers S&P 500 ESG ETF"),
                    ("SUSL", "iShares Trust - iShares ESG MSCI USA Leaders ETF"),
                    (
                        "EAGG",
                        "iShares Trust - iShares ESG Aware U.S. Aggregate Bond ETF",
                    ),
                ]
            )
        ]
    )[1],
}

savings_rate = 0.0

start_date = "2000-06-01 00:00:00"
end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

plt.rcParams["figure.figsize"] = [15, 5]
np.random.seed(42)

MAX = "MAX"
MIN = "MIN"

path = "./configs/profiles"
FILE = open(path, "r")
profiles = FILE.readlines()
FILE.close()
profiles = [profile.strip() for profile in profiles]

profiles_names = [
    "IRA",
    "Investment",
]
profiles_targets: List[Tuple[float, float]] = [
    (0.20, MAX),
    (0.50, MAX),
]
profiles_options: List[Set[str]] = [
    set(funds["ticks"]),
    set(funds["ticks"] + etfs["ticks"] + stocks["ticks"] + ["CASH"]),
]

ticks = funds["ticks"] + stocks["ticks"] + etfs["ticks"] + ["CASH"]

tick_names = funds["names"] + stocks["names"] + etfs["names"] + ["CASH"]

ticks, tick_names = [list(l) for l in zip(*sorted(zip(ticks, tick_names)))]


def get_weights(wts):
    wts = npg.maximum(0.0, wts)
    if np.sum(wts) != 0:
        wts = wts / np.sum(wts)
    return wts


def filterTickers(ticks, tick_allowed):
    return [tick for tick in ticks if tick in tick_allowed]


def convertWTS(weights, options):
    new_weights = np.zeros(len(ticks))
    index = 0
    for i in range(len(ticks)):
        if ticks[i] in options:
            new_weights[i] = weights[index]
            index += 1
        else:
            new_weights[i] = 0.0
    return get_weights(new_weights)


def options_to_matrix(options, ticks):
    matrix = np.zeros((len(options), len(ticks)))
    index = 0
    for i in range(len(ticks)):
        if ticks[i] in options:
            matrix[index, i] = 1.0
            index += 1
    return matrix


def ret_to_logret(ret):
    log_ret = np.log(ret + 1) / 252
    return log_ret


def logret_to_ret(log_ret):
    return np.exp(log_ret * 252) - 1


def get_return(wts, options):
    wts = convertWTS(np.array(wts), options)
    #     wts = get_weights(wts)
    port_ret = np.sum(log_ret_mean * wts)
    port_ret = np.exp(port_ret * 252) - 1
    return port_ret


def make_get_return(options, ticks):
    matrix = options_to_matrix(options, ticks)

    def get_return(wts, matrix=matrix):
        wts = wts @ matrix
        port_ret = np.sum(log_ret_mean * wts)
        port_ret = np.exp(port_ret * 252) - 1
        return port_ret

    return get_return


def get_risk(wts, options):
    wts = convertWTS(np.array(wts), options)
    port_sd = npg.sqrt(npg.dot(wts.T, npg.dot(cov_mat, wts)))
    return port_sd


def make_get_risk(options, ticks):
    matrix = options_to_matrix(options, ticks)

    def get_risk(wts, matrix=matrix):
        wts = wts @ matrix
        port_sd = npg.sqrt(np.dot(wts.T, np.dot(cov_mat, wts)))
        return port_sd

    return get_risk


def get_sharpe(wts, options):
    port_ret = get_return(wts, options)
    port_sd = get_risk(wts, options)
    sr = port_ret / port_sd
    return sr


def make_get_sharpe(options, ticks):
    get_risk = make_get_risk(options, ticks)
    get_return = make_get_return(options, ticks)

    def get_sharpe(wts, get_risk=get_risk, get_return=get_return):
        port_ret = get_return(wts)
        port_sd = get_risk(wts)
        sr = port_ret / port_sd
        return sr

    return get_sharpe


def get_weights_v(wts):
    wts = npg.maximum(0.0, wts)
    if npg.sum(wts) != 0:
        wts = wts / npg.sum(wts, axis=1, keepdims=True)
    return wts


def convertWTS_v(weights, options):
    new_weights = np.zeros((weights.shape[0], len(ticks)))
    index = 0
    for i in range(len(ticks)):
        if ticks[i] in options:
            new_weights[:, i] = weights[:, index]
            index += 1
        else:
            new_weights[:, i] = 0.0
    return get_weights_v(new_weights)


def get_return_v(wts, options):
    wts = convertWTS_v(wts, options)
    port_ret = npg.sum(log_ret_mean * wts, axis=1)
    port_ret = npg.exp(npg.multiply(port_ret, 252)) - 1
    return port_ret


def get_risk_v(wts, options):
    wts = convertWTS_v(wts, options)
    port_sd = npg.sqrt(npg.sum(wts * npg.dot(wts, cov_mat.T), axis=1))
    return port_sd


def get_sharpe_v(wts, options):
    port_ret = get_return_v(wts, options)
    port_sd = get_risk_v(wts, options)
    sr = port_ret / port_sd
    return sr


def get_weights_ratio(wts_1, wts_2, ratio):
    wts_1 = np.array(wts_1)
    wts_2 = np.array(wts_2)
    wts = wts_1 * ratio + wts_2 * (1.0 - ratio)
    wts = npg.maximum(0.0, wts)
    if npg.sum(wts) != 0:
        wts = wts / npg.sum(wts)
    return wts


def plot_risk_vs_return(risk, returns, risk_title="Risk", return_title="Return"):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(risk.T)
    ax[0].set_title(risk_title)
    ax[1].plot(returns.T)
    ax[1].set_title(return_title)

    ax[0].set_ylabel("Risk")
    ax[0].set_xlabel("Iteration")
    ax[1].set_ylabel("Return")
    ax[1].set_xlabel("Iteration")
    plt.show()


def plot_portfolio_composition(ticks, weights, plot_name, color_list, cash=None):
    x = dict()
    c = dict()
    for i in range(len(ticks)):
        if weights[i] >= 0.99:
            if cash:
                x[ticks[i]] = weights[i] * (cash)
                c[ticks[i]] = color_list[i]
                x[ticks[i] + " "] = weights[i] * (cash)
                c[ticks[i] + " "] = color_list[i]
            else:
                x[ticks[i]] = weights[i]
                c[ticks[i]] = color_list[i]
                x[ticks[i] + " "] = weights[i]
                c[ticks[i] + " "] = color_list[i]
        elif weights[i] > 0.001:
            if cash:
                x[ticks[i]] = weights[i] * (cash)
                c[ticks[i]] = color_list[i]
            else:
                x[ticks[i]] = weights[i]
                c[ticks[i]] = color_list[i]

    plot_data = (
        pd.Series(x).reset_index(name="value").rename(columns={"index": "stock"})
    )
    plot_data["angle"] = plot_data["value"] / plot_data["value"].sum() * 2 * math.pi

    plot_data["color"] = c.values()

    if cash:
        p = figure(
            width=50,
            height=50,
            title=plot_name,
            toolbar_location=None,
            sizing_mode="scale_height",
            tools="hover",
            tooltips="@stock: $@value{0,0.00} ",
            x_range=(-0.5, 0.5),
        )
    else:
        p = figure(
            width=50,
            height=50,
            title=plot_name,
            toolbar_location=None,
            sizing_mode="scale_height",
            tools="hover",
            tooltips="@stock: @value{%0.1f}",
            x_range=(-0.5, 0.5),
        )
    p.title.align = "center"
    p.wedge(
        x=0,
        y=1,
        radius=0.4,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        color="color",
        source=plot_data,
    )
    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    p.outline_line_color = None

    return p


path = "./configs/apikey"
FILE = open(path, "r")
api_key = FILE.readline()
FILE.close()

if "CASH" in ticks:
    ticks.remove("CASH")
if "CASH" in tick_names:
    tick_names.remove("CASH")

yf_data = yf.download(" ".join(ticks), period="5y")

price_data = yf_data["Close"].copy()
price_data["CASH"] = 1.0

log_ret = np.log(price_data / price_data.shift(1))

cov_mat = np.array(np.exp(log_ret.cov() * 252) - 1)
log_ret_mean = np.array(log_ret.mean(skipna=True))
log_ret_mean[-1] = ret_to_logret(savings_rate)

squared_diff = np.square(
    np.array(log_ret.mean(skipna=True)) - np.array(log_ret.median(skipna=True))
)
idx = np.argmax(squared_diff)

ticks = ticks + ["CASH"]
tick_names = tick_names + ["CASH"]

list_of_files = glob.glob(".\profiles\*.csv")

list_of_files
list_stamped = [
    (
        datetime.fromtimestamp(os.path.getctime(path)),
        pd.read_csv(path)[["Account Number", "Current Value"]]
        .replace({"\$": ""}, regex=True)
        .replace({"\,": ""}, regex=True)
        .replace({"\(": ""}, regex=True)
        .replace({"\)": ""}, regex=True)
        .astype({"Account Number": "str", "Current Value": "float"})
        .groupby("Account Number")
        .sum(),
    )
    for path in list_of_files
]
list_stamped = [(stamp, df[df.index.isin(profiles)]) for stamp, df in list_stamped]
value_over_time = pd.concat(
    [
        df.transpose().rename(index={"Current Value": stamp})
        for stamp, df in list_stamped
    ]
)
value_over_time.index = value_over_time.index.date
value_over_time.index.name = "Date"
value_over_time = value_over_time.sort_values("Date")

price_data_copy = price_data.copy()
price_data_copy.index = price_data_copy.index.date

price_data_with_profiles = price_data_copy.join(value_over_time)
price_data_with_profiles.index.name = "Date"

latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
df = pd.read_csv(latest_file)
df = df[~df.Description.isna() | (df.Symbol == "Pending Activity")]
df = df[["Account Number", "Symbol", "Current Value", "Last Price Change"]]

profile_makeup = pd.DataFrame(
    data=np.zeros((len(profiles), len(ticks))), index=profiles, columns=ticks
)


def clean(string):
    for char in "$,()":
        string = string.replace(char, "")
    return string


for row_ in df.iterrows():
    if row_[1]["Symbol"] in ticks and row_[1]["Account Number"] in profiles:
        profile_makeup.loc[row_[1]["Account Number"], row_[1]["Symbol"]] = float(
            clean(row_[1]["Current Value"])
        )
    elif row_[1]["Account Number"] in profiles:
        if (
            math.isnan(float(clean(str(row_[1]["Current Value"]))))
            and row_[1]["Symbol"] == "Pending Activity"
        ):
            profile_makeup.loc[row_[1]["Account Number"], "CASH"] += float(
                clean(str(row_[1]["Last Price Change"]))
            )
        else:
            profile_makeup.loc[row_[1]["Account Number"], "CASH"] += float(
                clean(str(row_[1]["Current Value"]))
            )

headers = {"Content-type": "application/json"}
data = json.dumps({"seriesid": ["CUUR0000SA0"], "startyear": "2020", "endyear": "2021"})
p = requests.post(
    "https://api.bls.gov/publicAPI/v2/timeseries/data/", data=data, headers=headers
)
json_data = json.loads(p.text)

df = pd.DataFrame.from_dict(json_data["Results"]["series"][0]["data"])
df.head()
values = df["value"].to_numpy().astype(float)
inflation = (values[:-12] - values[12:]) / values[12:]
current_inflation = inflation[0]
avg_inflation = inflation.mean()

best_sharpe_weights = []
for p_ in range(len(profiles_names)):
    get_sharpe_local = make_get_sharpe(profiles_options[p_], ticks)
    best_sharpe_weights.append(
        minimize(
            lambda x: -get_sharpe_local(x),
            np.random.random(len(profiles_options[p_])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ],
            bounds=[(0.0, 1.0) for i in range(len(profiles_options[p_]))],
        ).x
    )
    print(get_sharpe(best_sharpe_weights[p_], profiles_options[p_]))


min_risk_weights = []

for p_ in range(len(profiles_names)):
    get_return_local = make_get_return(profiles_options[p_], ticks)
    get_risk_local = make_get_risk(profiles_options[p_], ticks)
    get_sharpe_local = make_get_sharpe(profiles_options[p_], ticks)

    loss = make_get_risk(profiles_options[p_], ticks)

    min_risk_weights.append(
        minimize(
            loss,
            np.random.random(len(profiles_options[p_])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ],
            bounds=[(0.0, 1.0) for i in range(len(profiles_options[p_]))],
        ).x
    )
    print(get_risk(min_risk_weights[p_], profiles_options[p_]))


max_risk_weights = []

for p_ in range(len(profiles_names)):
    get_return_local = make_get_return(profiles_options[p_], ticks)
    get_risk_local = make_get_risk(profiles_options[p_], ticks)
    get_sharpe_local = make_get_sharpe(profiles_options[p_], ticks)

    loss = lambda x: -get_risk_local(x)

    max_risk_weights.append(
        minimize(
            loss,
            np.random.random(len(profiles_options[p_])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ],
            bounds=[(0.0, 1.0) for i in range(len(profiles_options[p_]))],
        ).x
    )
    print(get_risk(max_risk_weights[p_], profiles_options[p_]))


min_return_weights = []

for p_ in range(len(profiles_names)):
    get_return_local = make_get_return(profiles_options[p_], ticks)
    get_risk_local = make_get_risk(profiles_options[p_], ticks)
    get_sharpe_local = make_get_sharpe(profiles_options[p_], ticks)

    loss = get_return_local
    min_return_weights.append(
        minimize(
            loss,
            np.random.random(len(profiles_options[p_])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ],
            bounds=[(0.0, 1.0) for i in range(len(profiles_options[p_]))],
        ).x
    )
    print(get_return(min_return_weights[p_], profiles_options[p_]))


max_return_weights = []

for p_ in range(len(profiles_names)):
    get_return_local = make_get_return(profiles_options[p_], ticks)
    get_risk_local = make_get_risk(profiles_options[p_], ticks)
    get_sharpe_local = make_get_sharpe(profiles_options[p_], ticks)

    loss = lambda x: -get_return_local(x)
    max_return_weights.append(
        minimize(
            loss,
            np.random.random(len(profiles_options[p_])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ],
            bounds=[(0.0, 1.0) for i in range(len(profiles_options[p_]))],
        ).x
    )
    print(get_return(max_return_weights[p_], profiles_options[p_]))


risk_mins = [
    get_risk(min_risk_weights[p_], profiles_options[p_])
    for p_ in range(len(profiles_names))
]
risk_maxs = [
    get_risk(max_return_weights[p_], profiles_options[p_])
    for p_ in range(len(profiles_names))
]
risks_count = 50
risks_bot = [
    np.linspace(risk_min, risk_max, risks_count, True)
    for risk_min, risk_max in zip(risk_mins, risk_maxs)
]


best_weights_range = [
    np.random.random(size=(risks_count, len(profiles_options[p_])))
    for p_ in range(len(profiles_names))
]
start_time = time.time()
for p_ in range(len(profiles_names)):
    print("Running Profile {}".format(profiles_names[p_]))
    get_return_local = make_get_return(profiles_options[p_], ticks)
    get_risk_local = make_get_risk(profiles_options[p_], ticks)
    for r in range(risks_count):
        loss = lambda x: -get_return_local(x)
        rts = minimize(
            loss,
            np.random.random(len(profiles_options[p_])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {
                    "type": "ineq",
                    "fun": lambda w, risk=risks_bot[p_][r]: -(get_risk_local(w) - risk),
                },
            ],
            bounds=[(0.0, 1.0) for i in range(len(profiles_options[p_]))],
        )
        best_weights_range[p_][r, :] = rts.x
        print(rts.message)

        print("--- %s seconds ---" % (time.time() - start_time))


profiles_constraints = []
profiles_losses = []
for target, options in zip(profiles_targets, profiles_options):
    get_return_local = make_get_return(options, ticks)
    get_risk_local = make_get_risk(options, ticks)
    get_sharpe_local = make_get_sharpe(options, ticks)

    if target is None:
        profiles_constraints.append(None)
        profiles_losses.append(None)
    else:
        risk_t, return_t = target
        print(risk_t, return_t)
        if (risk_t == MAX or risk_t == MIN) and (return_t == MAX or return_t == MIN):
            profiles_constraints.append(None)
            if (risk_t == MAX) and (return_t == MAX):
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: -(
                        get_return_local(x) * get_risk_local(x)
                    )
                )
            elif (risk_t == MAX) and (return_t == MIN):
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: get_sharpe_local(
                        x
                    )
                )
            elif (risk_t == MIN) and (return_t == MAX):
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: -get_sharpe_local(
                        x
                    )
                )
            elif (risk_t == MIN) and (return_t == MIN):
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: get_return_local(
                        x
                    )
                    * get_risk_local(x)
                )
        elif risk_t == MAX or risk_t == MIN:
            print(return_t)
            profiles_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local, return_t=return_t: (
                        get_return_local(x) - return_t
                    ),
                }
            ),
            if risk_t == MAX:
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: -get_risk_local(
                        x
                    )
                )
            elif risk_t == MIN:
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: get_risk_local(
                        x
                    )
                )
        elif return_t == MAX or return_t == MIN:
            print(risk_t)
            profiles_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local, risk_t=risk_t: -(
                        get_risk_local(x) - risk_t
                    ),
                }
            ),
            if return_t == MAX:
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: -get_return_local(
                        x
                    )
                )
            elif return_t == MIN:
                profiles_losses.append(
                    lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local: get_return_local(
                        x
                    )
                )
        else:
            profiles_constraints.append(None)
            profiles_losses.append(
                lambda x, get_return_local=get_return_local, get_risk_local=get_risk_local, get_sharpe_local=get_sharpe_local, return_t=return_t, risk_t=risk_t: np.abs(
                    risk_t - get_risk_local(x)
                )
                * np.abs(return_t - get_return_local(x))
            )


b = 1000
i = 1000
lr = 0.015
batch = [b] * len(profiles)
iterations = [i] * len(profiles)
LR = [lr] * len(profiles)

target_weights = []

for i in range(len(profiles)):
    print(i)
    if profiles_targets[i] is not None:

        loss = profiles_losses[i]
        start_time = time.time()
        print(profiles_constraints[i])
        rts = minimize(
            loss,
            np.random.random(len(profiles_options[i])),
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                profiles_constraints[i],
            ]
            if profiles_constraints[i] is not None
            else [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            ],
            bounds=[(0.0, 1.0) for tick in range(len(profiles_options[i]))],
        )

        print(rts.success)
        target_weights.append(
            rts.x,
        )
        print(rts.x.shape)
        print(r)
        if profiles_constraints[i] is not None:
            print(0.1)
        print(get_risk(rts.x, profiles_options[i]))
        print(get_return(rts.x, profiles_options[i]))

        print("--- %s seconds ---" % (time.time() - start_time))

    else:

        wts = profile_makeup.loc[profiles[i]].to_numpy()
        target_weights.append(wts)


target_weights = [
    convertWTS(target, option)
    for target, option in zip(target_weights, profiles_options)
]


profile_changes = []
for i in range(len(profiles)):
    profile_sum = np.sum(profile_makeup.loc[profiles[i]].to_numpy())
    target_sum = np.sum(target_weights[i])
    changes = (
        target_weights[i] * (profile_sum / target_sum)
        - profile_makeup.loc[profiles[i]].to_numpy()
    )
    profile_changes.append(changes)


SAT = 0.5
LUM = 0.5


color_list = [
    colorsys.hls_to_rgb(h, LUM, SAT)
    for h in np.linspace(0.0, 1.0, len(ticks), endpoint=False)
]
color_list = [RGB(r * 255, g * 255, d * 255) for r, g, d in color_list]
color_list_accounts = [
    colorsys.hls_to_rgb(h, LUM, SAT)
    for h in np.linspace(0.0, 1.0, len(profiles), endpoint=False)
]
color_list_accounts = [
    RGB(r * 255, g * 255, d * 255) for r, g, d in color_list_accounts
]


tick_filter = np.zeros((len(ticks),))
for i in range(len(profiles)):
    tick_filter += profile_makeup.loc[profiles[i]].to_numpy()
    tick_filter += target_weights[i]
tick_filter = tick_filter > 0.001


SAT = 0.5
LUM = 0.5


color_list = [
    colorsys.hls_to_rgb(h, LUM, SAT)
    for h in np.linspace(0.0, 1.0, len(ticks), endpoint=False)
]
color_list = [RGB(r * 255, g * 255, d * 255) for r, g, d in color_list]
color_list_accounts = [
    colorsys.hls_to_rgb(h, LUM, SAT)
    for h in np.linspace(0.0, 1.0, len(profiles), endpoint=False)
]
color_list_accounts = [
    RGB(r * 255, g * 255, d * 255) for r, g, d in color_list_accounts
]


# ===== Setup Plot ====
p = figure(
    sizing_mode="stretch_both",
    title="Efficient frontier.",
    tools="box_zoom,wheel_zoom,reset",
    toolbar_location="right",
    x_range=(-0.25, 1.0),
)
p.add_tools(CrosshairTool(line_alpha=1, line_color="lightgray", line_width=1))
p.add_tools(HoverTool(tooltips=None))

p.xaxis.axis_label = "Volatility, or risk (standard deviation)"
p.yaxis.axis_label = "Annual return"
p.xaxis[0].formatter = NumeralTickFormatter(format="0.0%")
p.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
LINE_WIDTH = 3
# ===== Render Boundries ====
risk_boundry = Span(
    location=np.min(
        [
            get_risk(min_risk_wts, options)
            for min_risk_wts, options in zip(min_risk_weights, profiles_options)
        ]
    ),
    dimension="height",
    line_color="#3A5311",
    line_width=1,
)
return_boundry = Span(
    location=np.max(
        [
            get_return(max_return_wts, options)
            for max_return_wts, options in zip(max_return_weights, profiles_options)
        ]
    ),
    dimension="width",
    line_color="#3A5311",
    line_width=1,
)
current_inf_boundry = Span(
    location=current_inflation, dimension="width", line_color="#03C04A", line_width=1
)
average_inf_boundry = Span(
    location=avg_inflation, dimension="width", line_color="#607D3B", line_width=1
)
p.renderers.extend(
    [risk_boundry, return_boundry, current_inf_boundry, average_inf_boundry]
)
# ===== Render Best Sharpe Line ====
for p_ in range(len(profiles_names)):
    #     if p_ == 1: continue
    boundry = best_weights_range[p_]

    l = p.line(
        get_risk_v(boundry, profiles_options[p_]),
        get_return_v(boundry, profiles_options[p_]),
        color="purple",
        #     legend_label="Max Sharpe Line?",
        line_width=LINE_WIDTH,
    )

    p.add_tools(
        HoverTool(
            renderers=[l],
            tooltips=[("Name", "Max Sharpe Line {}".format(profiles_names[p_]))],
        )
    )
# ===== Render Sharpe Lines ====
max_ret = np.max(
    [
        get_return(max_return_wts, options)
        for max_return_wts, options in zip(max_return_weights, profiles_options)
    ]
)
p.line(
    [0, max_ret],
    [0, max_ret],
    #        legend_label="Sharpe Of 1",
    color="#00B7EB",
    line_width=LINE_WIDTH,
)
p.line(
    [0, 0.5 * max_ret],
    [0, max_ret],
    #        legend_label="Sharpe Of 2",
    color="#6495ED",
    line_width=LINE_WIDTH,
)
p.line(
    [0, (1.0 / 3.0) * max_ret],
    [0, max_ret],
    #        legend_label="Sharpe Of 3",
    color="#007FFF",
    line_width=LINE_WIDTH,
)
# ====== Render Prices ======
prices = figure(
    sizing_mode="stretch_both",
    title="Distance from Mean Over Time",
    tools="box_zoom,wheel_zoom,reset",
    toolbar_location="right",
    x_axis_type="datetime",
)

source = ColumnDataSource(price_data_with_profiles)
visible_range = 30
range_ = 365 * 1
prices.x_range.start = (
    datetime.now().timestamp() - visible_range * 24 * 60 * 60
) * 1000
prices.x_range.end = datetime.now().timestamp() * 1000
adjusted_prices = (price_data - price_data[-range_:].mean()) / price_data[
    -range_:
].std()
prices.y_range.start = adjusted_prices[-visible_range:].min().min()
prices.y_range.end = adjusted_prices[-visible_range:].max().max()

prices.add_tools(CrosshairTool(line_alpha=1, line_color="lightgray", line_width=1))
prices.add_tools(HoverTool(tooltips=None))

prices.xaxis.axis_label = "Date"
prices.yaxis.axis_label = "STD"
prices.xaxis[0].formatter = DatetimeTickFormatter(days=["%b %d, %Y"])
prices.yaxis[0].formatter = NumeralTickFormatter(format="0.0")

# ===== Plot prices =====
print(len(color_list) // sum(tick_filter))
source = ColumnDataSource(adjusted_prices)
renderers = []
tooltips = []
color_idx = 0
for i in range(len(ticks)):
    if tick_filter[i]:
        color_idx += 1
        l = prices.line(
            x="Date",
            y=ticks[i],
            source=source,
            legend_label=ticks[i],
            name=ticks[i],
            color=color_list[color_idx * ((len(color_list) - 1) // sum(tick_filter))],
            width=5,
        )
        renderers.append(l)
tooltips.append(("Ticker", "$name"))
prices.add_tools(HoverTool(renderers=renderers, tooltips=tooltips))
# ===== Render and plot accounts =====
account_values = figure(
    sizing_mode="stretch_both",
    title="Account totals Over Time",
    tools="box_zoom,wheel_zoom,reset",
    toolbar_location="right",
    x_axis_type="datetime"
    #     x_range=(-0.25,1.0)
)
# timestamp_start = (datetime.combine(datepicker_start.value, datetime.min.time())
#                         - datetime(1970, 1, 1)) / timedelta(seconds=1)
# timestamp_end = (datetime.combine(datepicker_end.value, datetime.min.time())
#                     - datetime(1970, 1, 1)) / timedelta(seconds=1)
source_profiles = ColumnDataSource(value_over_time)
visible_range = 90
range_ = 365 * 1
account_values.x_range.start = (
    datetime.now().timestamp() - visible_range * 24 * 60 * 60
) * 1000  # Multiply by 1e3 as JS timestamp is in milliseconds
account_values.x_range.end = (
    datetime.now().timestamp() * 1000
)  # Multiply by 1e3 as JS timestamp is in milliseconds
account_values.y_range.start = (
    price_data_with_profiles[-visible_range:].min().min() * 0.9
)
account_values.y_range.end = price_data_with_profiles[-visible_range:].max().max() * 1.1

account_values.add_tools(
    CrosshairTool(line_alpha=1, line_color="lightgray", line_width=1)
)
account_values.add_tools(HoverTool(tooltips=None))

account_values.xaxis.axis_label = "Date"
account_values.yaxis.axis_label = "STD"
account_values.xaxis[0].formatter = DatetimeTickFormatter(days=["%b %d, %Y"])
account_values.yaxis[0].formatter = NumeralTickFormatter(format="0.0")

renderers = []
tooltips = []
for i in range(len(profiles)):
    l = account_values.line(
        x="Date",
        y=profiles[i],
        source=source_profiles,
        legend_label=profiles[i],
        name=ticks[i],
        color=color_list[i * (len(color_list) // len(profiles))],
        width=5,
    )
    renderers.append(l)
tooltips.append(("Ticker", "$name"))
account_values.add_tools(HoverTool(renderers=renderers, tooltips=tooltips))

# ===== Render Rebalance Buy Charts ====
fidelity_buy_values = []
fidelity_buy_pies = []
for i in range(len(profiles)):

    if profiles_targets[i] is not None:
        wts = get_weights(profile_changes[i] * (profile_changes[i] > 0))
        cash = np.sum(profile_changes[i] * (profile_changes[i] > 0))
    else:
        wts = get_weights(profile_makeup.loc[profiles[i]].to_numpy())
        cash = np.sum(profile_makeup.loc[profiles[i]].to_numpy())
    if CASH_OVERIDE is not None:
        print(cash)
        cash = min(cash, CASH_OVERIDE)

    fidelity_buy_pies.append(
        plot_portfolio_composition(
            ticks, wts, profiles_names[i] + " Buy $%d" % (cash), color_list, cash=cash
        )
    )
    fidelity_buy_values.append(wts * cash)

print(fidelity_buy_pies)
# ===== Render Rebalance Sell Pie Charts ====
fidelity_sell_values = []
fidelity_sell_pies = []
for i in range(len(profiles)):

    if profiles_targets[i] is not None:
        wts = get_weights(-profile_changes[i] * (profile_changes[i] < 0))
        cash = np.sum(-profile_changes[i] * (profile_changes[i] < 0))

    else:
        wts = get_weights(profile_makeup.loc[profiles[i]].to_numpy())
        cash = np.sum(profile_makeup.loc[profiles[i]].to_numpy())
    if CASH_OVERIDE is not None:
        cash = min(cash, CASH_OVERIDE)

    fidelity_sell_pies.append(
        plot_portfolio_composition(
            ticks, wts, profiles_names[i] + " Sell $%d" % (cash), color_list, cash=cash
        )
    )
    fidelity_sell_values.append(wts * cash)

print(fidelity_sell_pies)
# ===== Render Target Profile Pie Charts ====
fidelity_targets = []
renderers = []

for i in range(len(target_weights)):
    fidelity_targets.append(
        plot_portfolio_composition(
            (ticks if len(target_weights[i]) > len(ticks) else ticks),
            get_weights(target_weights[i]),
            profiles_names[i] + " Target",
            color_list,
        )
    )
    if np.sum(target_weights[i]) != 0.0:
        c = p.circle(
            get_risk(get_weights(target_weights[i]), set(ticks)),
            get_return(get_weights(target_weights[i]), set(ticks)),
            color=color_list_accounts[i],
            alpha=0.6,
            name=profiles_names[i] + " Target",
            legend_label=profiles_names[i] + " Target",
            size=15,
        )
        renderers.append(c)
# ===== Render Existing Profile Pie Charts ====
print("===== Render Existing Profile Pie Charts ====")
fidelity_pies = []
tooltips = []
for i in range(len(profiles)):
    wts = get_weights(profile_makeup.loc[profiles[i]].to_numpy())
    print(wts)

    fidelity_pies.append(
        plot_portfolio_composition(ticks, wts, profiles_names[i], color_list)
    )
    if np.sum(wts) != 0.0:
        c = p.circle(
            get_risk(wts, set(ticks)),
            get_return(wts, set(ticks)),
            color=color_list_accounts[i],
            name=profiles_names[i],
            legend_label=profiles_names[i],
            size=15,
        )
        renderers.append(c)
print(fidelity_pies)
tooltips.append(("Profile", "$name"))
p.add_tools(HoverTool(renderers=renderers, tooltips=tooltips))
# ===== Render Funds ====
wts = np.eye(len(ticks))
risks_ = get_risk_v(wts, ticks)
returns_ = get_return_v(wts, ticks)
colors = color_list
print(
    "funds",
    list(
        zip(
            list(itertools.compress(risks_, tick_filter)),
            list(itertools.compress(returns_, tick_filter)),
        )
    ),
)
funds_source = ColumnDataSource(
    dict(
        risks=list(itertools.compress(risks_, tick_filter)),
        returns=list(itertools.compress(returns_, tick_filter)),
        color=list(itertools.compress(color_list, tick_filter)),
        ticks=list(itertools.compress(ticks, tick_filter)),
        tick_names=list(itertools.compress(tick_names, tick_filter)),
    )
)
renderers = []

renderers.append(
    p.circle(
        x="risks",
        y="returns",
        color="color",
        name="ticks",
        legend_field="ticks",
        size=10,
        alpha=0.8,
        source=funds_source,
    )
)
p.add_tools(
    HoverTool(
        renderers=renderers, tooltips=[("Tick", "@ticks"), ("Tick Name", "@tick_names")]
    )
)

# ===== Adjusting Legend ====
p.legend.location = "top_left"
# p.legend.visible = False
p.legend.click_policy = "hide"
p.legend.__setattr__("label_text_font_size", "8pt")

prices.legend.location = "top_left"
# t.legend.visible = False
prices.legend.click_policy = "hide"
prices.legend.__setattr__("label_text_font_size", "8pt")

account_values.legend.location = "top_left"
# t.legend.visible = False
account_values.legend.click_policy = "hide"
account_values.legend.__setattr__("label_text_font_size", "8pt")
# ===== Create dashboard and open new window to show results ====
names = [name + " Sell" for name in profiles_names] + [
    name + " Buy" for name in profiles_names
]
names_alt = [None] * (2 * len(profiles_names))
names_alt[::2] = [name + " Sell" for name in profiles_names]
names_alt[1::2] = [name + " Buy" for name in profiles_names]
DF = pd.DataFrame(
    {
        name: [round(val, 2) for val in vals]
        for name, vals in zip(names, fidelity_sell_values + fidelity_buy_values)
    },
    index=ticks,
)
DF = DF[DF.sum(axis=1) > 0]
cols = list(DF[DF.sum(axis=1) > 0].columns)
buy_sells_table = {col: list(DF[col]) for col in cols}
buy_sells_table["Ticks"] = list(DF.index)

buy_self_source = ColumnDataSource(buy_sells_table)
layout_ = row(
    [
        column([p, prices, account_values], sizing_mode="stretch_both"),
        column(
            [
                row(fidelity_pies, sizing_mode="stretch_height"),
                row(fidelity_targets, sizing_mode="stretch_height"),
                DataTable(
                    columns=[TableColumn(title="Ticks", field="Ticks")]
                    + [TableColumn(title=name, field=name) for name in names_alt],
                    index_position=None,
                    source=buy_self_source,
                    editable=True,
                    sizing_mode="stretch_both",
                ),
            ]
        ),
    ],
    #     width=1500,
    sizing_mode="stretch_both",
)

show(layout_)
