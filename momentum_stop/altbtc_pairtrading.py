import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from catalyst import run_algorithm
from catalyst.api import symbol, order_target_percent, record
from catalyst.exchange import exchange_errors as catalyst_errors
from catalyst.exchange.exchange_errors import CreateOrderError
from catalyst.exchange.utils.stats_utils import extract_transactions
from logbook import Logger, notifiers
import datetime

error_handler = notifiers.PushoverHandler(
    application_name="Binance",
    apikey="amrpo9ogp97wsak7zrf6k4uu1mbz5g",
    userkey="gqwfvx21h6rtv42m4bq826c23uoj43",
    level="CRITICAL",
    bubble=True,
)

NAMESPACE = "ALT_VSTOP"
log = Logger(NAMESPACE)
LIVE = True
BUY_ALL = False
LIQUIDATE_ALL = False


def initialize(context):
    for attempt in context.attempts:
        context.attempts[attempt] = 100

    context.set_commission(maker=0.003, taker=0.003)
    context.set_slippage(slippage=0.0005)
    context.first_run = True

    context.target = read_target()
    # context.set_benchmark(symbol('xrp_btc'))

    atr_length = 3
    context.assets = []
    atr_multiplier = range(1, 6)  # range of atr multis
    context.allocation = (1 / max(atr_multiplier)) / len(context.target.index)

    for pair in context.target.index:
        for atr in atr_multiplier:
            asset = CryptoAsset(symbol(pair), atr, atr_length, context.allocation, pair)
            context.assets.append(asset)
    print()


def handle_data(context, data):

    if execution_time() or not LIVE:
        log_portfolio(context, NAMESPACE)
        for asset in context.assets:
            load_history_and_indicators(context, data, asset)

            # log.info(f"Asset: {asset.symbol.asset_name} Price: {asset.price} VSTOP: {asset.vstop}")
            trigger_vstop_trend_change(context, asset)

        context.first_run = False
    if not LIVE:
        record(cash=context.portfolio.cash)
        if asset == context.assets[0]:
            record(
                price_0=asset.price,
                vstop_0=asset.vstop,
                amount=context.portfolio.positions[context.assets[0].symbol].amount,
            )


def execution_time():

    currentDT = datetime.datetime.now()
    print(f"{currentDT.hour} + {currentDT.minute}")
    if currentDT.hour == 17 and currentDT.minute == 41:
        return True
    else:
        return False


def load_history_and_indicators(context, data, asset):
    try:

        context.hist_len = 75  # 75 bars
        context.history_time = "1D"

        hist = data.history(asset.symbol, ["high", "low", "price"], 75, "1D")

        if context.first_run:

            print(f"Loading first history for {asset.pair} ")
            for Index, row in hist.iterrows():
                asset.atr = talib.ATR(
                    hist.iloc[: hist.index.get_loc(Index) + 1]["high"],
                    hist.iloc[: hist.index.get_loc(Index) + 1]["low"],
                    hist.iloc[: hist.index.get_loc(Index) + 1]["price"],
                    timeperiod=asset.atr_length,
                )[-1]
                asset.price = row["price"]
                asset.update_vstop()
                # print(f"price {asset.price}")
                # print(f"vstop {asset.vstop}")
                if not LIVE:
                    record(
                        cash=context.portfolio.cash,
                        price_0=asset.price,
                        vstop_0=asset.vstop,
                    )
        else:
            asset.atr = talib.ATR(
                hist["high"], hist["low"], hist["price"], timeperiod=asset.atr_length
            )[-1]

            asset.price = data.current(asset.symbol, "price")

            asset.update_vstop()

    except (
        ccxt.ExchangeError,
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        catalyst_errors.NoCandlesReceivedFromExchange,
    ) as error:
        log.critical(
            f"\nHistory retrieve error: {type(error)} \nArgs: {error.args} \n Asset: {asset.symbol}"
        )

        # skip calculation of asset and retry
        load_history_and_indicators(context, data, asset)


def trigger_vstop_trend_change(context, asset):
    if asset.trend_change:

        log.info(f"------------------------------------------------------------")
        log.info(
            f"Asset: {asset.symbol.asset_name} Price: {asset.price} VSTOP: {asset.vstop}"
        )

        # We check what's our position on our portfolio and trade accordingly
        if asset.uptrend:
            context.target.loc[[asset.pair][0]] += asset.portfolio_allocation
            asset.long = True
            try:
                order_target_percent(
                    asset=asset.symbol,
                    target=context.target.loc[asset.pair][0],
                    limit_price=asset.price * 1.01,
                )

            except CreateOrderError as e:
                log.critical(f"CreateOrderError {e.args}")

            log.info(
                f"1 Day Momstop: Trend changed BUY: {asset.symbol.asset_name} Price: {asset.price}"
            )

        elif not asset.uptrend:
            if context.portfolio.positions[asset.symbol].amount > 0 and asset.long:

                context.target.loc[[asset.pair][0]] -= asset.portfolio_allocation

                asset.long = False
                try:
                    order_target_percent(
                        asset=asset.symbol,
                        target=context.target.loc[asset.pair][0],
                        limit_price=asset.price * 0.99,
                    )
                    log.info(
                        f"1 Day: Trend changed SOLD: {asset.symbol.asset_name} Price: {asset.price}"
                    )
                except CreateOrderError as e:
                    log.critical(f"CreateOrderError {e.args}")
        save_target(context)

        log.info(f"------------------------------------------------------------")


def log_portfolio(context, name):
    # should reference self name
    log.critical(
        f"{name}"
        f"\nCash: {context.portfolio.cash} "
        # f"\nStarting Cash: {context.portfolio.starting_cash} "
        f"\nPortfolio value: {context.portfolio.positions_value} "
        f"\nProfit and Loss:: {context.portfolio.pnl} "
        f"\nReturn %: {round(context.portfolio.returns, 3)*100}"
    )


def read_target():
    target = pd.read_csv("target.csv", index_col=0)
    return target


def save_target(context):
    context.target.to_csv("target.csv")


class CryptoAsset(object):
    def __init__(self, coin, atr_multiplier, atr_length, portfolio_allocation, pair):
        self.symbol = coin
        self.pair = pair
        self.atr_multiplier = atr_multiplier
        self.atr_length = atr_length
        self.long = False
        self.portfolio_allocation = portfolio_allocation
        self.price = 0

        self.uptrend = True
        self.uptrend_prev = True
        self.trend_change = False
        self.vstop = 0
        self.max = 0
        self.min = 0
        self.atr = 0

    def update_vstop(self):

        self.uptrend_prev = self.uptrend

        # check last VSTOP values (OLD: value, vstop_vstop_is_uptrend_prev, uptrend, min, max) (NEW: atr)
        if self.uptrend_prev:
            temp_stop = max(
                self.vstop, max(self.max, self.price) - self.atr_multiplier * self.atr
            )
        else:
            temp_stop = min(
                self.vstop, min(self.min, self.price) + self.atr_multiplier * self.atr
            )

        self.uptrend = (self.price - temp_stop) >= 0
        self.trend_change = self.uptrend != self.uptrend_prev

        # Set new Max / Min
        self.max = self.price if self.trend_change else max(self.max, self.price)
        self.min = self.price if self.trend_change else min(self.min, self.price)

        # Set new value value
        if self.trend_change:
            if self.uptrend:
                self.vstop = self.max - self.atr_multiplier * self.atr
            else:
                self.vstop = self.min + self.atr_multiplier * self.atr
        else:
            self.vstop = temp_stop
        self.vstop = self.vstop


def analyze(context, perf):
    if not LIVE:
        # Get the base_currency that was passed as a parameter to the simulation
        exchange = list(context.exchanges.values())[0]

        print("Total return: " + str(perf.algorithm_period_return[-1] * 100))
        print("Sortino coef: " + str(perf.sortino[-1]))
        print("Max drawdown: " + str(np.min(perf.max_drawdown)))
        print("Alpha: " + str(perf.alpha[-1]))
        print("Beta: " + str(perf.beta[-1]))
        perf.to_csv("perf_" + str(context.assets[0].pair) + ".csv")

        base_currency = exchange.quote_currency.upper()
        fig = plt.figure(figsize=(18, 16), dpi=80, facecolor="w", edgecolor="k")

        # First chart: Plot portfolio value using base_currency
        ax1 = plt.subplot(611)
        perf.loc[:, ["portfolio_value"]].plot(ax=ax1)
        ax1.legend_.remove()
        ax1.set_ylabel("Portfolio Value\n({})".format(base_currency))
        start, end = ax1.get_ylim()
        ax1.xaxis.set_ticks(np.arange(start, end, (end - start) / 5))

        # Second chart: Plot asset price, value and buys/sells

        try:
            ax2 = plt.subplot(612, sharex=ax1)
            perf.loc[:, ["price_0", "vstop_0"]].plot(ax=ax2, label="Price + Vstop")
            ax2.legend_.remove()
            ax2.set_ylabel("Asset: \n({})".format(context.assets[0].pair))

            start, end = ax2.get_ylim()
            ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))

        except KeyError:
            log.info(f"SID: \n {KeyError} ")

        transaction_df = extract_transactions(perf)
        if not transaction_df.empty:
            try:
                transactions_0 = transaction_df["sid"] == context.assets[0].symbol
                transactions_0 = transaction_df.loc[transactions_0]
            except IndexError:
                log.info(f"SID: \n {IndexError} ")
            try:
                buy_df = transactions_0.loc[transactions_0["amount"] > 0]
                sell_df = transactions_0.loc[transactions_0["amount"] < 0]

                # temp hardcoded plotting should be based on # of assets
                ax2.scatter(
                    buy_df.index.to_pydatetime(),
                    perf.loc[buy_df.index, "price_0"],
                    marker="^",
                    s=100,
                    c="green",
                    label="",
                )
                ax2.scatter(
                    sell_df.index.to_pydatetime(),
                    perf.loc[sell_df.index, "price_0"],
                    marker="v",
                    s=100,
                    c="red",
                    label="",
                )
            except (KeyError, UnboundLocalError) as e:
                log.info(f"SID: \n {e} ")

        plt.show()


if __name__ == "__main__":
    if LIVE:
        run_algorithm(
            capital_base=4,
            data_frequency="minute",
            initialize=initialize,
            simulate_orders=True,
            handle_data=handle_data,
            exchange_name="binance",
            algo_namespace=NAMESPACE,
            quote_currency="btc",
            live=True,
        )
    else:
        run_algorithm(
            # How much cash
            capital_base=1000,
            data_frequency="daily",
            initialize=initialize,
            handle_data=handle_data,
            exchange_name="poloniex",
            analyze=analyze,
            algo_namespace=NAMESPACE,
            start=pd.to_datetime("2017-1-1", utc=True),
            end=pd.to_datetime("2018-12-24", utc=True),
            quote_currency="btc",
            live=False,
        )
