from trading.indicators import vstop
import ccxt
import talib
from catalyst.api import (order_target_percent)
from catalyst.exchange import exchange_errors as catalyst_errors
from catalyst.exchange.exchange_errors import CreateOrderError

from scrapped.strategy_template import StrategyTemplate


class PortfolioVstop(StrategyTemplate):

    def __init__(self, name, log, trading_pairs, execution_time, total_allocation, initial_leverage,
                 history_time_frame, quote_currency):
        super().__init__(name, log, trading_pairs, execution_time, total_allocation, initial_leverage,
                         history_time_frame, quote_currency)

        self.currently_trading = 0
        self.max_indicator_length = None
        self.atr_multiplier = None
        self.atr_length = None
        self.atr_length = None
        self.rsi_length = None
        self.rsi_trigger_level = None


        # Binance BTC specific variable
        self.btc_long = False

    def initialize_indicators(self, atr_multiplier, atr_length, rsi_length, rsi_trigger_level):
        self.max_indicator_length = rsi_length
        self.atr_multiplier = atr_multiplier
        self.atr_length = atr_length
        self.rsi_length = rsi_length
        self.rsi_trigger_level = rsi_trigger_level


        for asset in self.assets:
            asset.vstop = vstop.Vstop(atr_multiplier=atr_multiplier, uptrend=True)

    def handle_data(self, context, data):
        """
        Checks strategy time variable for execution called by handle data (only interface)
        # Load History
        #
        :param context: Received from handle data
        :param data: Received from handle data
        """

        self.timer += 1

        if self.timer >= self.execution_time:
            if self.currently_trading >= len(self.assets):
                self.timer = len(self.assets)
                return
        else:
            return

        if self.currently_trading == 0:
            self.log_portfolio(context)

        self.load_history( data, self.assets[self.currently_trading])
        self.load_indicators( data, self.assets[self.currently_trading])
        self.execution_logic(context, data, self.assets[self.currently_trading])

        # Cycle to next trading pair
        self.currently_trading += 1

    def load_history(self, data, asset):
        try:
            asset.history = data.history(asset.symbol,
                                         ['high', 'low', 'price'],
                                         self.max_indicator_length * 5,
                                         self.history_time_frame)
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout,
                catalyst_errors.NoCandlesReceivedFromExchange) as error:
            self.log.critical(f"\n{self.name} - History retrieve error: {type(error)}"
                              f"\n Args: {error.args} "
                              f"\n Asset: {asset.symbol}")

            # skip calculation of asset and retry
            self.currently_trading -= 1
            return

    def load_indicators(self, data, asset):
        if 'btc_usd' in asset.name:
            asset.rsi = talib.RSI(asset.history['price'],
                                  timeperiod=self.rsi_length)[-1]

        asset.atr = talib.ATR(asset.history['high'],
                              asset.history['low'],
                              asset.history['price'],
                              timeperiod=self.atr_length)[-1]

        asset.price = data.current(asset.symbol, 'price')

        asset.vstop.update_vstop(atr=asset.atr, price=asset.price)

    def execution_logic(self, context, data, asset):
        self.trigger_btc_rsi_dip(context, data, asset)
        self.trigger_vstop_trend_change(context, data, asset)

    def trigger_btc_rsi_dip(self, context, data, asset):
        if 'btc_usd' in asset.name:
            # RSI trigger
            if asset.rsi < self.rsi_trigger_level:
                # Portfolio all in on BTC RCI trigger (check history of last day?)
                self.log.warning(f"BTC RSI BUY Triggered - RSI {asset.rsi}")
                for coin in self.assets:
                    coin_amount = context.portfolio.positions[coin.symbol].amount
                    if coin_amount == 0:
                        coin_price = data.current(coin.symbol, 'price')
                        try:
                            if 'btc' in self.quote_currency and 'usd' in asset.name:
                                self.btc_long = True
                            else:
                                order_target_percent(asset=coin.symbol, target=coin.portfolio_allocation,
                                                     limit_price=coin_price * 1.02)
                        except CreateOrderError as e:
                            self.log.warning(f"CreateOrderError {e.args}")
                        coin.vstop_set_uptrend()

                        self.log.warning(f"Bought {coin.symbol} Confirmed")

    def trigger_vstop_trend_change(self, context, data, asset):
        if asset.vstop.vstop_trend_changed:
            # We check what's our position on our portfolio and trade accordingly
            if asset.vstop.vstop_is_uptrend:
                try:
                    # BTC based algo
                    if 'btc' in self.quote_currency:
                        if 'btc_usd' in asset.name:
                            self.btc_long = True
                        elif self.btc_long:
                            order_target_percent(asset=asset.symbol, target=asset.portfolio_allocation,
                                                 limit_price=asset.price * 1.02)
                    else:
                        order_target_percent(asset=asset.symbol, target=asset.portfolio_allocation,
                                             limit_price=asset.price * 1.02)

                    self.log.critical(f"Trend changed BUY: {asset.symbol} Price: {asset.price}")
                except CreateOrderError as e:
                    self.log.critical(f"CreateOrderError {e.args}")

            elif not asset.vstop_is_uptrend:
                try:
                    if 'btc' in self.quote_currency:
                        if 'btc_usd' in asset.name:
                            self.btc_long = False
                        else:
                            order_target_percent(asset=asset.symbol, target=0, limit_price=asset.price * 0.98)
                            if not context.btc_long:
                                self.log.critical(f"Sold asset {asset.symbol} requires BTC dump to USDT")
                    else:
                        order_target_percent(asset=asset.symbol, target=0, limit_price=asset.price * 0.98)

                    self.log.critical(f"Trend changed SELL: {asset.symbol} Price: {asset.price}")
                except CreateOrderError as e:
                    self.log.critical(f"CreateOrderError {e.args}")

    def record(self):
        pass

    def analyze(self):
        pass
