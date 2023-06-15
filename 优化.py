import itertools
import talib
import numpy as np
import pandas as pd
import yfinance as yf
import random
import csv
from deap import base, creator, tools, algorithms
from sklearn.feature_selection import SelectKBest, f_regression
import itertools
from tqdm import tqdm


# 选择了重要的33个indicators做
indicators = [
    ('SMA', lambda x, timeperiod=30: talib.SMA(x, timeperiod=timeperiod)),
    ('EMA', lambda x, timeperiod=30: talib.EMA(x, timeperiod=timeperiod)),
    ('WMA', lambda x, timeperiod=30: talib.WMA(x, timeperiod=timeperiod)),
    ('MACD', lambda x, fastperiod=12, slowperiod=26, signalperiod=9: talib.MACD(
        x, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0]),
    ('RSI', lambda x, timeperiod=14: talib.RSI(x, timeperiod=timeperiod)),
    ('BBANDS', lambda x, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0: talib.BBANDS(
        x, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0]),
    ('ADX', lambda high, low, close, timeperiod=14: talib.ADX(high, low, close, timeperiod=timeperiod)),
    ('STOCH', lambda high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0:
    talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                slowd_period=slowd_period, slowd_matype=slowd_matype)[0]),
    ('CCI', lambda high, low, close, timeperiod=14: talib.CCI(high, low, close, timeperiod=timeperiod)),
    ('ROC', lambda x, timeperiod=10: talib.ROC(x, timeperiod=timeperiod)),
    ('ATR', lambda high, low, close, timeperiod=14: talib.ATR(high, low, close, timeperiod=timeperiod)),
    ('OBV', lambda close, volume: talib.OBV(close, volume)),
    ('MFI', lambda high, low, close, volume, timeperiod=14: talib.MFI(high, low, close, volume, timeperiod=timeperiod)),
    ('AD', talib.AD),
    ('WILLR', lambda high, low, close, timeperiod=14: talib.WILLR(high, low, close, timeperiod=timeperiod)),
    ('ULTOSC', lambda high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28: talib.ULTOSC(
        high, low, close, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)),
    ('SAR', lambda high, low, acceleration=0.02, maximum=0.2: talib.SAR(high, low, acceleration=acceleration, maximum=maximum)),
    ('STDDEV', lambda x, timeperiod=5, nbdev=1: talib.STDDEV(x, timeperiod=timeperiod, nbdev=nbdev)),
    ('NATR', lambda high, low, close, timeperiod=14: talib.NATR(high, low, close, timeperiod=timeperiod)),
    ('TRANGE', lambda high, low, close: talib.TRANGE(high, low, close)),
    ('ADOSC', lambda high, low, close, volume, fastperiod=3, slowperiod=10: talib.ADOSC(
        high, low, close, volume, fastperiod=fastperiod, slowperiod=slowperiod)),
    ('AVGPRICE', lambda open, high, low, close: talib.AVGPRICE(open, high, low, close)),
    ('MEDPRICE', lambda high, low: talib.MEDPRICE(high, low)),
    ('TYPPRICE', lambda high, low, close: talib.TYPPRICE(high, low, close)),
    ('WCLPRICE', lambda high, low, close: talib.WCLPRICE(high, low, close)),
    ('HT_TRENDLINE', lambda x: talib.HT_TRENDLINE(x)),
    ('KAMA', lambda x, timeperiod=30: talib.KAMA(x, timeperiod=timeperiod)),
    ('TEMA', lambda x, timeperiod=30: talib.TEMA(x, timeperiod=timeperiod)),
    ('HT_DCPERIOD', lambda x: talib.HT_DCPERIOD(x)),
    ('HT_DCPHASE', lambda x: talib.HT_DCPHASE(x)),
    ('HT_PHASOR', lambda x: talib.HT_PHASOR(x)[0]),
    ('HT_SINE', lambda x: talib.HT_SINE(x)[0]),
    ('HT_TRENDMODE', lambda x: talib.HT_TRENDMODE(x)),
]

indicators_dict = dict(indicators)

default_params = {
    'SMA': {'timeperiod': 30},
    'EMA': {'timeperiod': 30},
    'WMA': {'timeperiod': 30},
    'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
    'RSI': {'timeperiod': 14},
    'BBANDS': {'timeperiod': 5, 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0},
    'ADX': {'timeperiod': 14},
    'STOCH': {'fastk_period': 5, 'slowk_period': 3, 'slowk_matype': 0, 'slowd_period': 3, 'slowd_matype': 0},
    'CCI': {'timeperiod': 14},
    'ROC': {'timeperiod': 10},
    'ATR': {'timeperiod': 14},
    'OBV': {},
    'MFI': {'timeperiod': 14},
    'AD': {},
    'WILLR': {'timeperiod': 14},
    'ULTOSC': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
    'SAR': {'acceleration': 0.02, 'maximum': 0.2},
    'STDDEV': {'timeperiod': 5, 'nbdev': 1},
    'NATR': {'timeperiod': 14},
    'TRANGE': {},
    'ADOSC': {'fastperiod': 3, 'slowperiod': 10},
    'AVGPRICE': {},
    'MEDPRICE': {},
    'TYPPRICE': {},
    'WCLPRICE': {},
    'HT_TRENDLINE': {},
    'KAMA': {'timeperiod': 30},
    'TEMA': {'timeperiod': 30},
    'HT_DCPERIOD': {},
    'HT_DCPHASE': {},
    'HT_PHASOR': {},
    'HT_SINE': {},
    'HT_TRENDMODE': {}
}

params_ranges = {
    'SMA': {'timeperiod': list(range(5, 61, 5))},
    'EMA': {'timeperiod': list(range(5, 61, 5))},
    'WMA': {'timeperiod': list(range(5, 61, 5))},
    'MACD': {'fastperiod': list(range(5, 16, 3)), 'slowperiod': list(range(17, 31, 3)), 'signalperiod': list(range(6, 15, 3))},
    'RSI': {'timeperiod': list(range(5, 31, 3))},
    'BBANDS': {'timeperiod': list(range(5, 31, 3)), 'nbdevup': list(np.arange(1, 3, 0.6)), 'nbdevdn': list(
        np.arange(1, 3, 0.6)), 'matype': list(range(0, 9, 3))},
    'ADX': {'timeperiod': list(range(5, 31, 3))},
    'STOCH': {'fastk_period': list(range(5, 21, 3)), 'slowk_period': list(range(3, 15, 3)), 'slowk_matype': list(range(0, 9, 3)),
              'slowd_period': list(range(3, 15, 3)), 'slowd_matype': list(range(0, 9, 3))},
    'CCI': {'timeperiod': list(range(5, 31, 3))},
    'ROC': {'timeperiod': list(range(5, 21, 3))},
    'ATR': {'timeperiod': list(range(5, 31, 3))},
    'OBV': {},
    'MFI': {'timeperiod': list(range(5, 31, 3))},
    'AD': {},
    'WILLR': {'timeperiod': list(range(5, 31, 3))},
    'ULTOSC': {'timeperiod1': list(range(5, 16, 3)), 'timeperiod2': list(range(17, 31, 3)), 'timeperiod3': list(range(32, 61, 3))},
    'SAR': {'acceleration': list(np.arange(0.01, 0.21, 0.03)), 'maximum': list(np.arange(0.1, 0.5, 0.15))},
    'STDDEV': {'timeperiod': list(range(5, 21, 3)), 'nbdev': list(np.arange(1, 3, 0.6))},
    'NATR': {'timeperiod': list(range(5, 31, 3))},
    'TRANGE': {},
    'ADOSC': {'fastperiod': list(range(2, 15, 3)), 'slowperiod': list(range(16, 31, 3))},
    'AVGPRICE': {},
    'MEDPRICE': {},
    'TYPPRICE': {},
    'WCLPRICE': {},
    'HT_TRENDLINE': {},
    'KAMA': {'timeperiod': list(range(5, 61, 5))},
    'TEMA': {'timeperiod': list(range(5, 61, 5))},
    'HT_DCPERIOD': {},
    'HT_DCPHASE': {},
    'HT_PHASOR': {},
    'HT_SINE': {},
    'HT_TRENDMODE': {}
}



def calculate_indicators(df, indicators):
    result = pd.DataFrame(index=df.index)
    for name, func in indicators:
        try:
            if name in ['ADX', 'CCI', 'ATR', 'WILLR', 'ULTOSC','STOCH','NATR', 'TRANGE', 'TYPPRICE', 'WCLPRICE']:
                result[name] = func(df['High'], df['Low'], df['Close'])
            elif name in ['OBV']:
                result[name] = func(df['Close'], df['Volume'])
            elif name in ['MFI', 'AD','ADOSC']:
                result[name] = func(df['High'], df['Low'], df['Close'], df['Volume'])
            elif name == 'AVGPRICE':
                result[name] = func(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
            elif name in ['SAR','MEDPRICE']:
                result[name] = func(df['High'], df['Low'])
            else:
                result[name] = func(df['Close'])
        except Exception as e:
            print(f'Error computing {name}: {str(e)}')


    # 添加两个新的指标
    high, low = df['High'].max(), df['Low'].min()
    diff = high - low
    result['FIB38'] = high - 0.382 * diff
    result['FIB62'] = high - 0.618 * diff
    return result


# 计算单独的指标（用于优化参数）
def calculate_indicator(df, indicator, params):
    result = pd.DataFrame(index=df.index)

    name = indicator
    func = indicators_dict[name]

    try:
        if name in ['ADX', 'CCI', 'ATR', 'WILLR', 'ULTOSC', 'STOCH', 'NATR', 'TRANGE', 'TYPPRICE', 'WCLPRICE']:
            result[name] = func(df['High'], df['Low'], df['Close'], **params)
        elif name in ['OBV']:
            result[name] = func(df['Close'], df['Volume'], **params)
        elif name in ['MFI', 'AD', 'ADOSC']:
            result[name] = func(df['High'], df['Low'], df['Close'], df['Volume'], **params)
        elif name == 'AVGPRICE':
            result[name] = func(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values,
                                **params)
        elif name in ['SAR', 'MEDPRICE']:
            result[name] = func(df['High'], df['Low'], **params)
        else:
            result[name] = func(df['Close'], **params)
    except Exception as e:
        print(f'Error computing {name}: {str(e)}')

    return result


def simulate_trading(df, indicator_values, target, initial_cash):
    cash = initial_cash
    shares = 0
    buy_threshold = indicator_values.quantile(0.75)
    sell_threshold = indicator_values.quantile(0.25)

    portfolio_values = []
    max_value = 0
    max_drawdown = 0
    returns = []
    number_of_trades = 0

    for i in range(len(df)):
        if indicator_values.iloc[i].values[0] > buy_threshold.values[0]:
            if cash >= df[target][i]:
                cash -= df[target][i]
                shares += 1
                number_of_trades += 1
        elif indicator_values.iloc[i].values[0] < sell_threshold.values[0]:
            if shares > 0:
                cash += df[target][i]
                shares -= 1
                number_of_trades += 1

        current_value = cash + shares * df[target][i]  # Fixed this line
        portfolio_values.append(current_value)

        if i > 0:
            returns.append((current_value - portfolio_values[i - 1]) / portfolio_values[i - 1])

        max_value = max(max_value, current_value)
        drawdown = (max_value - current_value) / max_value
        max_drawdown = max(max_drawdown, drawdown)

    total_value = cash + shares * df[target].iloc[-1]

    # Calculate the annualized volatility at the end
    annualized_volatility = np.std(returns) * np.sqrt(252)

    # Including returns in the return statement
    return cash, shares, total_value, max_drawdown, annualized_volatility, number_of_trades, returns




def select_best_indicators(df, target, indicators, k=3):
    best_indicators = {}
    initial_cash = 10000

    for indicator in indicators:
        indicator_values = calculate_indicators(df, [indicator])
        indicator_values = indicator_values.shift()  # to avoid lookahead bias

        # 处理缺失
        indicator_values.fillna(0, inplace=True)

        cash, shares, total_value, max_drawdown, ann_volatility, number_of_trades, returns = simulate_trading(
            df,indicator_values,target,initial_cash)

        # 计算投资收益
        profit = total_value - initial_cash

        # 计算夏普比率

        avg_returns = np.mean(returns) * 252  # Assume 252 trading days一年
        std_returns = np.std(returns) * np.sqrt(252)
        # print(std_returns)
        risk_free_rate = 0
        if std_returns == 0:
            print("Standard Deviation of Returns is zero, can't calculate Sharpe Ratio.")
            sharpe_ratio = -1e5  # 一个很小的值，用于方便计算score

        elif np.isnan(avg_returns) or np.isnan(std_returns):
            print("Encountered NaN values, can't calculate Sharpe Ratio.")
            sharpe_ratio = -1e5  # 一个很小的值

        else:
            sharpe_ratio = (avg_returns - risk_free_rate) / std_returns


        # 计算综合得分
        score = sharpe_ratio / (max_drawdown + 0.0001)

        best_indicators[indicator[0]] = {
            'Profit': profit,
            'Max Drawdown': max_drawdown,
            'Annualized Volatility': ann_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Score': score,
        }

    # 排序，选出最佳的k个指标
    best_indicators = dict(sorted(best_indicators.items(), key=lambda item: item[1]['Score'], reverse=True)[:k])
    return best_indicators


def find_best_indicators_for_all_stocks(all_data, target, indicators, k=3):
    best_indicators_dict = {}

    for stock_code, df in all_data.items():

        best_indicators = select_best_indicators(df, target, indicators, k)
        best_indicators_dict[stock_code] = best_indicators


    return best_indicators_dict




def calculate_sharpe_ratio(returns):
    # Assuming risk free rate is 0
    return np.mean(returns) / np.std(returns)


# 定义优化函数
def optimize_parameters(params_ranges, data_all, target='Close', initial_cash=10000):
    all_results = []

    # Loop 每个stock
    for stock_code, df in data_all.items():
        # 对每一个指标进行循环
        for indicator, params in tqdm(params_ranges.items()):

            best_score = float('-inf')
            best_param = None
            best_performance = None
            best_trades = 0

            # 如果该指标没有需要优化的参数
            if not params:
                indicator_values = calculate_indicator(df, indicator, {})
                indicator_values = indicator_values.shift()
                indicator_values.fillna(0, inplace=True)

                cash, shares, total_value, max_drawdown, ann_volatility, number_of_trades, returns = simulate_trading(
                    df, indicator_values, target, initial_cash)

                # 计算夏普比率
                avg_returns = np.mean(returns) * 252  # Assuming 252 trading days in a year
                std_returns = np.std(returns) * np.sqrt(252)
                risk_free_rate = 0  # You can change this value based on actual data
                sharpe_ratio = (avg_returns - risk_free_rate) / std_returns

                # 计算profit
                profit = total_value - initial_cash

                score = sharpe_ratio / (max_drawdown + 0.0001)

                all_results.append({
                    'Stock Code': stock_code,
                    'Indicator': indicator,
                    'Params': {},
                    'Score': score,
                    'Profit': profit,
                    'Max Drawdown': max_drawdown,
                    'Annualized Volatility': ann_volatility,
                    'Sharpe Ratio': sharpe_ratio,
                    'Number of Trades': number_of_trades
                })
                continue

            # 对于每一个参数进行循环
            for param in itertools.product(*params.values()):
                param_dict = dict(zip(params.keys(), param))

                # 计算指标值
                indicator_values = calculate_indicator(df, indicator, param_dict)
                indicator_values = indicator_values.shift()
                indicator_values.fillna(0, inplace=True)

                # 评估参数效果
                cash, shares, total_value, max_drawdown, ann_volatility, number_of_trades, returns = simulate_trading(
                    df, indicator_values, target, initial_cash)

                # 计算夏普比率
                avg_returns = np.mean(returns) * 252  # Assuming 252 trading days in a year
                std_returns = np.std(returns) * np.sqrt(252)
                risk_free_rate = 0  # You can change this value based on actual data
                sharpe_ratio = (avg_returns - risk_free_rate) / std_returns

                # 计算profit
                profit = total_value - initial_cash

                score = sharpe_ratio / (max_drawdown + 0.0001)

                # 如果参数的得分更高则替代
                if score > best_score:
                    best_score = score
                    best_param = param_dict
                    best_trades = number_of_trades
                    best_performance = {
                        'Profit': profit,
                        'Max Drawdown': max_drawdown,
                        'Annualized Volatility': ann_volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'Number of Trades': number_of_trades
                    }

            # 保存最佳参数的结果
            all_results.append({
                'Stock Code': stock_code,
                'Indicator': indicator,
                'Params': best_param,
                'Score': best_score,
                'Profit': best_performance['Profit'],
                'Max Drawdown': best_performance['Max Drawdown'],
                'Annualized Volatility': best_performance['Annualized Volatility'],
                'Sharpe Ratio': best_performance['Sharpe Ratio'],
                'Number of Trades': best_trades
            })

    # 储存为csv文件
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('best_params.csv', index=False)


def optimize_annualized_return(params_ranges, data_all, target='Close', initial_cash=10000):
    all_results = []

    # Loop 每个stock
    for stock_code, df in data_all.items():

        # 对每一个指标进行循环
        for indicator, params in tqdm(params_ranges.items()):

            best_score = float('-inf')
            best_param = None
            best_performance = None
            best_trades = 0

            # 对于每一个参数进行循环
            for param in itertools.product(*params.values()):
                param_dict = dict(zip(params.keys(), param))

                # 计算指标值
                indicator_values = calculate_indicator(df, indicator, param_dict)
                indicator_values = indicator_values.shift()
                indicator_values.fillna(0, inplace=True)

                # 评估参数效果
                cash, shares, total_value, max_drawdown, ann_volatility, number_of_trades, returns = simulate_trading(
                    df, indicator_values, target, initial_cash)

                # 计算累计收益
                cumulative_returns = np.prod([1 + r for r in returns]) - 1

                # 计算年化收益
                number_of_years = len(returns) / 252
                annualized_return = (1 + cumulative_returns) ** (1 / number_of_years) - 1

                # 用年化收益作为得分
                if number_of_trades == 0:
                    score = 0
                else:
                    score = annualized_return

                # 如果参数的得分更高则替代
                if score > best_score:
                    best_score = score
                    best_param = param_dict
                    best_trades = number_of_trades
                    best_performance = {
                        'Profit': total_value - initial_cash,
                        'Max Drawdown': max_drawdown,
                        'Annualized Volatility': ann_volatility,
                        'Annualized Return': annualized_return,
                        'Number of Trades': number_of_trades
                    }

            # 保存最佳参数的结果
            all_results.append({
                'Stock Code': stock_code,
                'Indicator': indicator,
                'Params': best_param,
                'Score': best_score,
                'Profit': best_performance['Profit'],
                'Max Drawdown': best_performance['Max Drawdown'],
                'Annualized Volatility': best_performance['Annualized Volatility'],
                'Annualized Return': best_performance['Annualized Return'],
                'Number of Trades': best_trades
            })

    # 储存为csv文件
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='Score', ascending=False)
    results_df.to_csv('best_params_annualized_return.csv', index=False)



def optimize_sharpe_ratio(params_ranges, data_all, target='Close', initial_cash=10000):
    all_results = []
    # Loop 每个stock
    for stock_code, df in data_all.items():
        # 对每一个指标进行循环
        for indicator, params in tqdm(params_ranges.items()):

            best_score = float('-inf')
            best_param = None
            best_performance = None
            best_trades = 0

            # 对于每一个参数进行循环
            for param in itertools.product(*params.values()):
                param_dict = dict(zip(params.keys(), param))

                # 计算指标值
                indicator_values = calculate_indicator(df, indicator, param_dict)
                indicator_values = indicator_values.shift()
                indicator_values.fillna(0, inplace=True)

                # 评估参数效果
                cash, shares, total_value, max_drawdown, ann_volatility, number_of_trades, returns = simulate_trading(
                    df, indicator_values, target, initial_cash)

                # 计算夏普比率
                avg_returns = np.mean(returns) * 252  # Assuming 252 trading days in a year
                std_returns = np.std(returns) * np.sqrt(252)
                risk_free_rate = 0  # You can change this value based on actual data
                if std_returns != 0 and not np.isnan(std_returns):
                    sharpe_ratio = (avg_returns - risk_free_rate) / std_returns
                else:
                    sharpe_ratio = -1e5

                # 用夏普比率作为得分
                if number_of_trades == 0:
                    score = 0
                else:
                    score = sharpe_ratio

                # 如果参数的得分更高则替代
                if score > best_score:
                    best_score = score
                    best_param = param_dict
                    best_trades = number_of_trades
                    best_performance = {
                        'Profit': total_value - initial_cash,
                        'Max Drawdown': max_drawdown,
                        'Annualized Volatility': ann_volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'Number of Trades': number_of_trades
                    }

            # 保存最佳参数的结果
            all_results.append({
                'Stock Code': stock_code,
                'Indicator': indicator,
                'Params': best_param,
                'Score': best_score,
                'Profit': best_performance['Profit'],
                'Max Drawdown': best_performance['Max Drawdown'],
                'Annualized Volatility': best_performance['Annualized Volatility'],
                'Sharpe Ratio': best_performance['Sharpe Ratio'],
                'Number of Trades': best_trades
            })

    # 储存为csv文件
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='Score', ascending=False)
    results_df.to_csv('best_params_sharpe_ratio.csv', index=False)


def optimize_annualized_return_max_drawdown_ratio(params_ranges, data_all, target='Close', initial_cash=10000):
    all_results = []

    # Loop 每个stock
    for stock_code, df in data_all.items():

        # 对每一个指标进行循环
        for indicator, params in tqdm(params_ranges.items()):

            best_score = float('-inf')
            best_param = None
            best_performance = None
            best_trades = 0

            # 对于每一个参数进行循环
            for param in itertools.product(*params.values()):
                param_dict = dict(zip(params.keys(), param))

                # 计算指标值
                indicator_values = calculate_indicator(df, indicator, param_dict)
                indicator_values = indicator_values.shift()
                indicator_values.fillna(0, inplace=True)

                # 评估参数效果
                cash, shares, total_value, max_drawdown, ann_volatility, number_of_trades, returns = simulate_trading(
                    df, indicator_values, target, initial_cash)

                # 计算累计收益
                cumulative_returns = np.prod([1 + r for r in returns]) - 1

                # 计算年化收益
                number_of_years = len(returns) / 252
                annualized_return = (1 + cumulative_returns) ** (1 / number_of_years) - 1

                # 用年化收益作为得分
                if number_of_trades == 0:
                    score = 0
                else:
                    score = annualized_return/(0.0001 + max_drawdown)

                # 如果参数的得分更高则替代
                if score > best_score:
                    best_score = score
                    best_param = param_dict
                    best_trades = number_of_trades
                    best_performance = {
                        'Profit': total_value - initial_cash,
                        'Max Drawdown': max_drawdown,
                        'Annualized Volatility': ann_volatility,
                        'Annualized Return': annualized_return,
                        'Number of Trades': number_of_trades
                    }

            # 保存最佳参数的结果
            all_results.append({
                'Stock Code': stock_code,
                'Indicator': indicator,
                'Params': best_param,
                'Score': best_score,
                'Profit': best_performance['Profit'],
                'Max Drawdown': best_performance['Max Drawdown'],
                'Annualized Volatility': best_performance['Annualized Volatility'],
                'Annualized Return': best_performance['Annualized Return'],
                'Number of Trades': best_trades
            })

    # 储存为csv文件
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='Score', ascending=False)
    results_df.to_csv('best_params_annualized_return_max_drawdown_ratio.csv', index=False)


def print_hi(name):

    # 下载的沪深300表格，找到沪深300的股票代码
    df = pd.read_excel('/Users/lizan/Downloads/000300cons.xls')

    stocks = df['成分券代码Constituent Code'].unique()

    yahoo_stocks = []
    yahoo_stocks.append('000300.SS')

    # 更改股票代码的格式为调用yahoo finance准备
    for stock in stocks:
        stock_str = str(stock).zfill(6)  # 保留0
        if stock_str.startswith('6'):
            yahoo_stock = stock_str + '.SS'
        else:
            yahoo_stock = stock_str + '.SZ'
        yahoo_stocks.append(yahoo_stock)

    data_all = {}

    for stock in yahoo_stocks[:1]:
        data = yf.download(stock, start='2018-05-26', end='2023-05-26')
        data_all[stock] = data

    # 处理target中的缺失数据
    for stock_code, df in data_all.items():
        df['target'] = df['Close'].dropna()
        df = df.dropna()
        data_all[stock_code] = df


    all_3data_best_indicators = find_best_indicators_for_all_stocks(data_all, 'target', indicators)
    print(all_3data_best_indicators)
    best_indicator_dict = {stock: max(indicators_dict, key=lambda x: indicators_dict[x]['Score']) for
                           stock, indicators_dict in all_3data_best_indicators.items()}

    # 最佳指标
    print(best_indicator_dict)
    # print(indicators)

    # # 优化所有参数
    # optimize_parameters(params_ranges, data_all)

    # 按照annualized return优化参数
    optimize_annualized_return(params_ranges, data_all)

    #
    optimize_annualized_return_max_drawdown_ratio(params_ranges, data_all)

    optimize_sharpe_ratio(params_ranges, data_all)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

