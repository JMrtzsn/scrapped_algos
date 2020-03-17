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

NAMESPACE = "binance_t1"
log = Logger(NAMESPACE)
LIVE = True
BUY_ALL = False
LIQUIDATE_ALL = False


def initialize(context):
    for attempt in context.attempts:
        context.attempts[attempt] = 100

    context.trading_pairs = [
        "btc_usdt",
        "eth_usdt",
        "ltc_usdt",
        "eos_usdt",
        "xrp_usdt",
        "iota_usdt",
        "bch_usdt",
        "trx_usdt",
        "xlm_usdt",
        "ada_usdt",
    ]

    context.allocation = 0.5 / 9
    atr_length = 12
    atr_multiplier = 2
    context.rsi_length = 14
    dip_levels = 31.5
    context.assets = []
    context.day_timer = 1441

    context.btc_long = False

    context.trading_execution_time = 1440
    context.history_time = "1D"

    context.buy_all = BUY_ALL
    context.liquidate_all = LIQUIDATE_ALL

    context.first_run = True

    if LIVE:
        error_handler.push_application()
    # context.exchanges['bitfinex'].get_balances()

    # Create BTC seperatly
    asset = CryptoAsset(symbol("btc_usdt"), dip_levels, atr_multiplier, atr_length, 0.5)
    context.assets.append(asset)

    for index, pair in enumerate(context.trading_pairs[1:]):
        asset = CryptoAsset(
            symbol(pair), dip_levels, atr_multiplier, atr_length, context.allocation
        )
        context.assets.append(asset)


def handle_data(context, data):
    context.day_timer += 1

    if LIVE:
        log.info(f" Current day timer: {context.day_timer} ")

    if context.day_timer > context.trading_execution_time:

        log_portfolio(context, NAMESPACE)
        for asset in context.assets:

            # ------History / Indicator logic------ #
            load_history_and_indicators(context, data, asset)
            # ------ History / Indicator logic------ #

            # ------ Order logic ------ #
            # Portfolio triggers:
            if context.buy_all:
                trigger_buy_holdings(context, data)
                context.buy_all = False
            if context.liquidate_all:
                trigger_liquidate_holdings(context, data)
                context.liquidate_all = False

            log.info(
                f"Asset: {asset.symbol.asset_name} Price: {asset.price} VSTOP: {asset.vstop}"
            )

            # Specific assets triggers:
            trigger_btc_rsi_dip(context, data, asset)
            trigger_vstop_trend_change(context, data, asset)
            # ------ Order logic ------ #

            # ------ Record Graph variables ------#
            if not LIVE:
                record(cash=context.portfolio.cash)
                if asset == context.assets[0]:
                    record(price_0=asset.price, vstop_0=asset.vstop, rsi=asset.rsi)
                if asset == context.assets[1]:
                    record(price_1=asset.price, vstop_1=asset.vstop)
                if asset == context.assets[2]:
                    record(price_2=asset.price, vstop_2=asset.vstop)

        context.day_timer = 1
        context.first_run = False


def load_history_and_indicators(context, data, asset):
    try:
        if context.assets[0].symbol == asset.symbol:

            # Log portfolio values to pushover once every hour

            hist = data.history(
                asset.symbol,
                ["high", "low", "price"],
                context.rsi_length * 5,
                context.history_time,
            )

            asset.rsi = talib.RSI(hist["price"], timeperiod=context.rsi_length)[-1]
        else:
            hist = data.history(
                asset.symbol,
                ["high", "low", "price"],
                asset.atr_length * 4,
                context.history_time,
            )

        asset.atr = talib.ATR(
            hist["high"], hist["low"], hist["price"], timeperiod=asset.atr_length
        )[-1]

        asset.price = data.current(asset.symbol, "price")
        if context.first_run:
            asset.vstop_set_downtrend()

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


def trigger_btc_rsi_dip(context, data, asset):
    # RSI trigger
    if context.assets[0].symbol == asset.symbol:
        if asset.rsi < asset.dip_level:
            # Portfolio all in on BTC RCI trigger (check history of last day?)
            log.critical(f"4 Hour Momstop: BTC RSI BUY Triggered - RSI {asset.rsi}")
            for coin in context.assets:
                coin_amount = context.portfolio.positions[coin.symbol].amount
                coin_price = data.current(coin.symbol, "price")
                if context.assets[0].symbol == coin.symbol:
                    context.btc_long = True
                if (
                    context.btc_long
                    and coin_amount * coin_price
                    < (asset.portfolio_allocation * context.portfolio.portfolio_value)
                    * 0.4
                ):
                    try:
                        order_target_percent(
                            asset=coin.symbol,
                            target=coin.portfolio_allocation,
                            limit_price=coin_price * 1.2,
                        )
                    except CreateOrderError as e:
                        log.critical(f"CreateOrderError {e.args}")
                    coin.vstop_set_uptrend()

                    log.critical(
                        f"1 DAY Momstop: Bought {coin.symbol.asset_name} Confirmed"
                    )


def trigger_vstop_trend_change(context, data, asset):
    pos_amount = context.portfolio.positions[asset.symbol].amount

    if asset.vstop_trend_changed:
        log.critical(
            f"Asset: {asset.symbol.asset_name} Price: {asset.price} VSTOP: {asset.vstop}"
        )
        # We check what's our position on our portfolio and trade accordingly
        if asset.vstop_is_uptrend:
            # if this is BTCUSDT skip and set to long
            if context.assets[0].symbol == asset.symbol:
                context.btc_long = True
            if (
                context.btc_long
                and pos_amount * asset.price
                < (asset.portfolio_allocation * context.portfolio.portfolio_value) * 0.4
            ):
                try:
                    order_target_percent(
                        asset=asset.symbol,
                        target=asset.portfolio_allocation,
                        limit_price=asset.price * 1.2,
                    )
                except CreateOrderError as e:
                    log.critical(f"CreateOrderError {e.args}")

            log.critical(
                f"1 Day Momstop: Trend changed BUY: {asset.symbol.asset_name} Price: {asset.price}"
            )

        elif not asset.vstop_is_uptrend:
            # if this is BTCUSDT skip and set to long
            if context.assets[0].symbol == asset.symbol:
                context.btc_long = False
            if pos_amount > 0:
                try:
                    order_target_percent(
                        asset=asset.symbol, target=0, limit_price=asset.price * 0.8
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
                asset=coin.symbol, target=0, limit_price=coin_price * 0.9
            )
            log.critical(f"Sold {coin.symbol.asset_name}")
        except CreateOrderError as e:
            log.critical(f"CreateOrderError {e.args}")


def trigger_buy_holdings(context, data):
    for index, coin in enumerate(context.assets):
        coin_price = data.current(coin.symbol, "price")
        try:
            if (
                context.portfolio.positions[coin.symbol].amount * coin_price
                < coin.portfolio_allocation * context.portfolio.portfolio_value
            ):
                order_target_percent(
                    asset=coin.symbol,
                    target=coin.portfolio_allocation,
                    limit_price=coin_price * 1.2,
                )
                log.critical(f"Bought {coin.symbol.asset_name}")
                coin.vstop_set_uptrend()
        except CreateOrderError as e:
            log.critical(f"CreateOrderError {e.args}")


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

    def rsi_dip_exit(self, price):
        self.rsi_vstop_sell = self.vstop
        if price >= self.rsi_vstop_sell > 0 and not self.vstop_is_uptrend:
            # Sell
            return True
        else:
            # Do nothing
            return False

    def vstop_set_uptrend(self):
        self.vstop_is_uptrend = True
        self.vstop = self.price - self.atr_mult * self.atr
        self.vstop_min = self.price
        self.vstop_min = self.price
        self.vstop_max = self.price
        self.vstop = round(self.vstop, 2)

    def vstop_set_downtrend(self):
        self.vstop_is_uptrend = False
        self.vstop = self.price + self.atr_mult * self.atr
        self.vstop_min = self.price
        self.vstop_max = self.price
        self.vstop = round(self.vstop, 2)

    def update_vstop(self):

        self.vstop_is_uptrend_prev = self.vstop_is_uptrend

        # check last VSTOP values (OLD: value, vstop_vstop_is_uptrend_prev, uptrend, min, max) (NEW: atr)
        if self.vstop_is_uptrend_prev:
            temp_stop = max(
                self.vstop, max(self.vstop_max, self.price) - self.atr_mult * self.atr
            )
        else:
            temp_stop = min(
                self.vstop, min(self.vstop_min, self.price) + self.atr_mult * self.atr
            )

        self.vstop_is_uptrend = (self.price - temp_stop) >= 0
        self.vstop_trend_changed = self.vstop_is_uptrend != self.vstop_is_uptrend_prev

        # Set new Max / Min
        self.vstop_max = (
            self.price if self.vstop_trend_changed else max(self.vstop_max, self.price)
        )
        self.vstop_min = (
            self.price if self.vstop_trend_changed else min(self.vstop_min, self.price)
        )

        # Set new value value
        if self.vstop_trend_changed:
            if self.vstop_is_uptrend:
                self.vstop = self.vstop_max - self.atr_mult * self.atr
            else:
                self.vstop = self.vstop_min + self.atr_mult * self.atr
        else:
            self.vstop = temp_stop
        self.vstop = round(self.vstop, 2)


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

            # Second chart: Plot asset price, value and buys/sells
            ax3 = plt.subplot(613, sharex=ax1)
            perf.loc[:, ["price_1", "vstop_1"]].plot(ax=ax3, label="Price + Vstop")
            ax3.legend_.remove()
            ax3.set_ylabel("Asset: \n({})".format(context.trading_pairs[1]))

            start, end = ax3.get_ylim()
            ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))

            ax4 = plt.subplot(614, sharex=ax1)
            perf.loc[:, ["price_2", "vstop_2"]].plot(ax=ax4, label="Price + Vstop")
            ax4.legend_.remove()
            ax4.set_ylabel("Asset: \n({})".format(context.trading_pairs[2]))

            start, end = ax4.get_ylim()
            ax4.yaxis.set_ticks(np.arange(start, end, (end - start) / 10))
        except KeyError:
            log.info(f"SID: \n {KeyError} ")

        transaction_df = extract_transactions(perf)
        if not transaction_df.empty:
            try:
                transactions_0 = transaction_df["sid"] == context.assets[0].symbol
                transactions_0 = transaction_df.loc[transactions_0]
                transactions_1 = transaction_df["sid"] == context.assets[1].symbol
                transactions_1 = transaction_df.loc[transactions_1]
                transactions_2 = transaction_df["sid"] == context.assets[2].symbol
                transactions_2 = transaction_df.loc[transactions_2]
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

                buy_df = transactions_1.loc[transactions_1["amount"] > 0]
                sell_df = transactions_1.loc[transactions_1["amount"] < 0]
                ax3.scatter(
                    buy_df.index.to_pydatetime(),
                    perf.loc[buy_df.index, "price_1"],
                    marker="^",
                    s=100,
                    c="green",
                    label="",
                )
                ax3.scatter(
                    sell_df.index.to_pydatetime(),
                    perf.loc[sell_df.index, "price_1"],
                    marker="v",
                    s=100,
                    c="red",
                    label="",
                )

                buy_df = transactions_2.loc[transactions_2["amount"] > 0]
                sell_df = transactions_2.loc[transactions_2["amount"] < 0]
                ax4.scatter(
                    buy_df.index.to_pydatetime(),
                    perf.loc[buy_df.index, "price_2"],
                    marker="^",
                    s=100,
                    c="green",
                    label="",
                )
                ax4.scatter(
                    sell_df.index.to_pydatetime(),
                    perf.loc[sell_df.index, "price_2"],
                    marker="v",
                    s=100,
                    c="red",
                    label="",
                )
            except (KeyError, UnboundLocalError) as e:
                log.info(f"SID: \n {e} ")

        ax5 = plt.subplot(615, sharex=ax1)
        perf.loc[:, "rsi"].plot(ax=ax5, label="RSI")
        ax5.set_ylabel("RSI")
        start, end = ax5.get_ylim()
        ax5.yaxis.set_ticks(np.arange(0, end, end / 5))

        plt.show()


if __name__ == "__main__":
    if LIVE:
        run_algorithm(
            capital_base=1000,
            data_frequency="minute",
            initialize=initialize,
            simulate_orders=True,
            handle_data=handle_data,
            exchange_name="binance",
            algo_namespace=NAMESPACE,
            quote_currency="usdt",
            live=True,
        )
    else:
        run_algorithm(
            # How much cash
            capital_base=1000,
            data_frequency="minute",
            initialize=initialize,
            handle_data=handle_data,
            exchange_name="binance",
            analyze=analyze,
            algo_namespace=NAMESPACE,
            start=pd.to_datetime("2018-6-1", utc=True),
            end=pd.to_datetime("2018-6-30", utc=True),
            quote_currency="usd",
            live=False,
        )
