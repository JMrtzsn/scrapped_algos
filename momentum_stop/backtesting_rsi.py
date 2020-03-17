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

error_handler = notifiers.PushoverHandler(
    application_name="Binance",
    apikey="amrpo9ogp97wsak7zrf6k4uu1mbz5g",
    userkey="gqwfvx21h6rtv42m4bq826c23uoj43",
    level="CRITICAL",
    bubble=True,
)

NAMESPACE = "backtesting"
log = Logger(NAMESPACE)
LIVE = False


def initialize(context):
    for attempt in context.attempts:
        context.attempts[attempt] = 100

    context.trading_pairs = ["btc_usdt"]
    context.allocation = 1
    atr_length = 3
    atr_multiplier = 4
    context.rsi_length = 28
    dip_levels = 1
    context.assets = []
    context.maxPortfolio = 0
    context.btc_long = False

    context.trading_execution_time = 1440
    context.history_time = "1D"

    context.first_run = True

    for index, pair in enumerate(context.trading_pairs):
        asset = CryptoAsset(
            symbol(pair), dip_levels, atr_multiplier, atr_length, context.allocation
        )
        context.assets.append(asset)


def handle_data(context, data):

    log_portfolio(context, NAMESPACE)
    for asset in context.assets:

        load_history_and_indicators(context, data, asset)

        trigger_rsi_dip(context, data, asset)
        trigger_vstop_trend_change(context, data, asset)
        trigger_loss_check(context, data)

        if not LIVE:
            record(cash=context.portfolio.cash)
            if asset == context.assets[0]:
                record(price_0=asset.rsi, vstop_0=asset.vstop, rsi=asset.price)

    context.first_run = False


def load_history_and_indicators(context, data, asset):
    try:

        hist = data.history(
            asset.symbol,
            ["high", "low", "price"],
            context.rsi_length * 5,
            context.history_time,
        )

        asset.rsi = talib.RSI(hist["price"], timeperiod=context.rsi_length)[-1]

        asset.rsi_dip = talib.RSI(hist["price"], timeperiod=14)[-1]

        asset.atr = talib.ATR(
            hist["high"], hist["low"], hist["price"], timeperiod=asset.atr_length
        )[-1]

        asset.price = data.current(asset.symbol, "price")
        if context.first_run:
            asset.vstop_set_downtrend(asset.rsi)

        asset.update_vstop(asset.rsi)

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


def trigger_rsi_dip(context, data, asset):
    if asset.rsi_dip < asset.dip_level:

        pos_amount = context.portfolio.positions[asset.symbol].amount

        if (
            pos_amount * asset.price
            < (asset.portfolio_allocation * context.portfolio.portfolio_value) * 0.5
        ):
            try:
                order_target_percent(
                    asset=asset.symbol,
                    target=asset.portfolio_allocation,
                    limit_price=asset.price * 1.5,
                )
            except CreateOrderError as e:
                log.critical(f"CreateOrderError {e.args}")

            asset.vstop_set_uptrend(asset.rsi)


def trigger_vstop_trend_change(context, data, asset):
    pos_amount = context.portfolio.positions[asset.symbol].amount

    if asset.vstop_trend_changed:
        log.critical(
            f"Asset: {asset.symbol.asset_name} Price: {asset.price} VSTOP: {asset.vstop}"
        )
        # We check what's our position on our portfolio and trade accordingly
        if asset.vstop_is_uptrend:
            if (
                pos_amount * asset.price
                < (asset.portfolio_allocation * context.portfolio.portfolio_value) * 0.5
            ):
                try:
                    order_target_percent(
                        asset=asset.symbol,
                        target=asset.portfolio_allocation,
                        limit_price=asset.price * 1.5,
                    )
                except CreateOrderError as e:
                    log.critical(f"CreateOrderError {e.args}")

            log.critical(
                f"1 Day Momstop: Trend changed BUY: {asset.symbol.asset_name} Price: {asset.price}"
            )

        elif not asset.vstop_is_uptrend:
            if pos_amount > 0:
                try:
                    order_target_percent(
                        asset=asset.symbol, target=0, limit_price=asset.price * 0.9
                    )
                    log.critical(
                        f"1 Day: Trend changed SOLD: {asset.symbol.asset_name} Price: {asset.price}"
                    )
                except CreateOrderError as e:
                    log.critical(f"CreateOrderError {e.args}")


def trigger_liquidate_holdings(context, data):
    log.critical(f"Liquidate all triggered")
    for coin in context.assets:
        coin_price = data.current(coin.symbol, "price")
        try:
            order_target_percent(
                asset=coin.symbol, target=0, limit_price=coin_price * 0.5
            )
            log.critical(f"Sold {coin.symbol.asset_name}")
            coin.vstop_set_downtrend(coin.rsi)
        except CreateOrderError as e:
            log.critical(f"CreateOrderError {e.args}")


def trigger_loss_check(context, data):
    current_portfolio = round(context.portfolio.returns, 3) * 100

    if context.maxPortfolio < current_portfolio:
        context.maxPortfolio = current_portfolio

    if round(context.portfolio.returns, 3) * 100 <= context.maxPortfolio - 15:
        trigger_liquidate_holdings(context, data)
        context.maxPortfolio = current_portfolio


def log_portfolio(context, name):
    # should reference self name
    log.critical(
        f"{name}"
        f"\nCash: {context.portfolio.cash} "
        f"\nStarting Cash: {context.portfolio.starting_cash} "
        f"\nPortfolio value: {context.portfolio.positions_value} "
        f"\nProfit and Loss:: {context.portfolio.pnl} "
        f"\nReturn %: {round(context.portfolio.returns, 3)*100}"
    )


class CryptoAsset(object):
    def __init__(
        self, symbol, dip_level, atr_multiplier, atr_length, portfolio_allocation
    ):
        self.symbol = symbol
        self.dip_level = dip_level
        self.atr_mult = atr_multiplier
        self.rsi = 0
        self.atr_length = atr_length
        self.portfolio_allocation = portfolio_allocation
        # Add RSI

        # Vstop variables
        self.vstop_is_uptrend = True
        self.vstop_is_uptrend_prev = True
        self.vstop_trend_changed = False
        self.vstop = 0
        self.vstop_max = 0
        self.vstop_min = 0
        self.atr = 0
        self.price = 0
        self.rsi_vstop_sell = 0

        self.step_value = 7

    def rsi_dip_exit(self, price):
        self.rsi_vstop_sell = self.vstop
        if price >= self.rsi_vstop_sell > 0 and not self.vstop_is_uptrend:
            # Sell
            return True
        else:
            # Do nothing
            return False

    def vstop_set_uptrend(self, input):
        self.vstop_is_uptrend = True
        self.vstop = input - self.step_value
        self.vstop_min = input
        self.vstop_min = input
        self.vstop_max = input
        self.vstop = self.vstop

    def vstop_set_downtrend(self, input):
        self.vstop_is_uptrend = False
        self.vstop = input + self.step_value
        self.vstop_min = input
        self.vstop_max = input
        self.vstop = self.vstop

    def update_vstop(self, input):

        self.vstop_is_uptrend_prev = self.vstop_is_uptrend

        if self.vstop_is_uptrend_prev:
            temp_stop = max(self.vstop, max(self.vstop_max, input) - self.step_value)
        else:
            temp_stop = min(self.vstop, min(self.vstop_min, input) + self.step_value)

        self.vstop_is_uptrend = (input - temp_stop) >= 0
        self.vstop_trend_changed = self.vstop_is_uptrend != self.vstop_is_uptrend_prev

        # Set new Max / Min
        self.vstop_max = (
            input if self.vstop_trend_changed else max(self.vstop_max, input)
        )
        self.vstop_min = (
            input if self.vstop_trend_changed else min(self.vstop_min, input)
        )

        # Set new value value
        if self.vstop_trend_changed:
            if self.vstop_is_uptrend:
                self.vstop = self.vstop_max - self.step_value
            else:
                self.vstop = self.vstop_min + self.step_value
        else:
            self.vstop = temp_stop
        self.vstop = self.vstop


def analyze(context, perf):
    if not LIVE:
        # Get the base_currency that was passed as a parameter to the simulation
        exchange = list(context.exchanges.values())[0]

        log.info(
            "Total Profit: \n {} ".format(perf.loc[:, ["portfolio_value"]].iloc[-1])
        )

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
            ax2.set_ylabel("Asset: \n({})".format(context.trading_pairs[0]))

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

                ax3 = plt.subplot(615, sharex=ax1)
                perf.loc[:, "rsi"].plot(ax=ax3, label="RSI")
                ax3.set_ylabel("RSI")
                start, end = ax3.get_ylim()
                ax3.yaxis.set_ticks(np.arange(0, end, end / 5))

            except (KeyError, UnboundLocalError) as e:
                log.info(f"SID: \n {e} ")

        plt.show()


if __name__ == "__main__":
    if LIVE:
        run_algorithm(
            capital_base=1000,
            data_frequency="minute",
            initialize=initialize,
            simulate_orders=True,
            handle_data=handle_data,
            exchange_name="bitfinex",
            algo_namespace=NAMESPACE,
            quote_currency="usd",
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
            start=pd.to_datetime("2016-1-1", utc=True),
            end=pd.to_datetime("2018-10-1", utc=True),
            quote_currency="usd",
            live=False,
        )
