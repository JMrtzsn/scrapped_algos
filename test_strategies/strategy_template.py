from catalyst.api import symbol, order_target_percent
from catalyst.exchange.exchange_errors import CreateOrderError

from scrapped.crypto_asset import CryptoAsset


class StrategyTemplate(object):
    def __init__(
        self,
        name,
        log,
        trading_pairs,
        execution_time,
        total_allocation,
        initial_leverage,
        history_time_frame,
        quote_currency,
    ):
        """
        Base init function, same for all strategies.
        :param trading_pairs: Dict of symbol and allocation.
        :param execution_time: Which execution time this algo should function.
        :param total_allocation: How much of total % Portfolio USDT to use for strategy.
        :param initial_leverage: Initial % of total_allocation to trade
        """
        self.log = log
        self.assets = []
        self.execution_time = execution_time
        self.leverage = initial_leverage
        self.total_allocation = total_allocation
        self.name = name
        self.history_time_frame = history_time_frame
        self.indicators = {}
        self.quote_currency = quote_currency

        for pair, asset_allocation in trading_pairs.items():
            asset = CryptoAsset(
                name=pair,
                symbol=symbol(pair),
                portfolio_allocation=asset_allocation
                * total_allocation
                * initial_leverage,
            )
            self.assets.append(asset)

        self.timer = execution_time
        self.execution_time = execution_time

    def __str__(self):
        return self.name

    def initialize(self):
        """
        Initialize the strategies specific variables
        :return:
        """

    def handle_data(self, context, data):
        """
        Checks strategy time variable for execution called by handle data (only interface)
        # Load History
        #
        :param context: Received from handle data
        :param data: Received from handle data
        """

    def load_history(self, context, data, asset):
        """
        Loads history
        initalizes all required indicators to a list? should be used in the same way after init is done
        :return:
        """

    def load_indicators(self, context, data, asset):
        """
        Loads Indicator history
        initalizes all required indicators to a list? should be used in the same way after init is done
        :return:
        """

    def execution_logic(self, context, data, asset):
        """

        UNIQUE TO STRATEGY OVERRIDE
        Execution logic goes here called from on tick trigger
        :return:
        """

    def record(self,):
        pass

    def analyze(self, context, perf):
        pass

    def sell_all_holdings(self, context, data):
        self.log.warning(f"{self.name} - Sell all triggered")
        for asset in context.assets:
            asset_price = data.current(asset.symbol, "price")
            try:
                order_target_percent(
                    asset=asset.symbol, target=0, limit_price=asset_price * 0.98
                )
                self.log.warning(f"{self.name} - Sold: {asset.symbol}")
            except CreateOrderError as e:
                self.log.warning(f"{self.name} - CreateOrderError {e.args}")

    def buy_all_holdings(self, context, data):
        self.log.warning(f"{self.name} - Buy all triggered")
        for asset in context.assets:
            asset_price = data.current(asset.symbol, "price")
            try:
                order_target_percent(
                    asset=asset.symbol,
                    target=asset.portfolio_allocation,
                    limit_price=asset_price * 1.02,
                )
                self.log.warning(f"{self.name} - Bought {asset.symbol}")
                asset.vstop_set_uptrend()
            except CreateOrderError as e:
                self.log.warning(f"{self.name} - CreateOrderError {e.args}")

    def set_holdings(self, context, data, asset):
        for asset in context.assets:
            if asset.old_portfolio_allocation < asset.portfolio_allocation:
                # we are selling some assets
                limit_price = asset.price * 1.02
            else:
                # we are buying assets
                limit_price = asset.price * 0.98
            try:

                order_target_percent(
                    asset=asset.symbol,
                    target=asset.portfolio_allocation,
                    limit_price=limit_price,
                )

                self.log.warning(
                    f"{self.name} - Set_Holding triggered {asset.symbol},"
                    f"\n{asset.new_portfolio_allocation} to {asset.new_portfolio_allocation} "
                )

                # what do?
                # asset.vstop_set_uptrend()
            except CreateOrderError as e:
                self.log.warning(f"{self.name} - CreateOrderError {e.args}")

    def log_portfolio(self, context):
        # should reference self name
        self.log.warning(
            f"{self.name}"
            f"\nPortfolio value: {context.portfolio.portfolio_value} "
            f"\nProfit and Loss:: {context.portfolio.pnl} "
            f"\nReturn %: {round(context.portfolio.returns, 3) * 100}"
        )
