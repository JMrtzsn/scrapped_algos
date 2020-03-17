import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target, order_target_percent, get_datetime)
from logbook import Logger, notifiers
import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from catalyst import run_algorithm
from catalyst.api import (symbol, order_target_percent, record)
from catalyst.exchange import exchange_errors as catalyst_errors
from catalyst.exchange.exchange_errors import CreateOrderError
from catalyst.exchange.utils.stats_utils import extract_transactions
from logbook import Logger, notifiers

from logbook import Logger, notifiers

from trading.indicators.pyrenko import renko

log = Logger('Testing_Renko')

def initialize(context):
    context.asset = symbol('btc_usdt')
    context.leverage = 1.0  # 1.0 - no leverage
    context.n_history = 24*15  # Number of lookback bars for modelling
    context.tf = '60T'  # How many minutes in a timeframe
    context.model = renko()  # Renko object
    context.part_cover_ratio = 0.0  # Partially cover position ratio
    context.last_brick_size = 0.0  # Last optimal brick size (just for storing)

    context.set_benchmark(context.asset)
    context.set_commission(maker=0.003, taker=0.003)
    context.set_slippage(slippage=0.0005)


# Function for optimization
def evaluate_renko(brick, history, column_name):
    renko_obj = renko()
    renko_obj.set_brick_size(brick_size=brick, auto=False)
    renko_obj.build_history(prices=history)
    return renko_obj.evaluate()[column_name]


def handle_data(context, data):
    current_time = get_datetime().time()
    if current_time.hour == 0 and current_time.minute == 0:
        print('Current date is ' + str(get_datetime().date()))

    # we check if the model is empty we should get the data,
    # calculate IQR, optimize the brick size, build the Renko chart, and open the order.
    if len(context.model.get_renko_prices()) == 0:
        context.model = renko()
        history = data.history(context.asset,
                               'price',
                               bar_count=context.n_history,
                               frequency=context.tf
                               )

        # Get daily absolute returns
        diffs = history.diff(1).abs()
        diffs = diffs[~np.isnan(diffs)]
        # Calculate Interquartile range of daily returns
        iqr_diffs = np.percentile(diffs, [25, 75])

        # Find the optimal brick size
        opt_bs = opt.fminbound(lambda x: -evaluate_renko(brick=x,
                                                         history=history, column_name='score'),
                               iqr_diffs[0], iqr_diffs[1], disp=0)

        # Build the model
        print('REBUILDING RENKO: ' + str(opt_bs))
        context.last_brick_size = opt_bs
        context.model.set_brick_size(brick_size=opt_bs, auto=False)
        context.model.build_history(prices=history)

        # Open a position
        if context.model.get_renko_directions()[-1] == 1:
            order_target_percent(context.asset, 1)
        elif context.model.get_renko_directions()[-1] == -1:
            order_target_percent(context.asset, 0)

        # Open a position
        order_target_percent(context.asset, context.leverage * context.model.get_renko_directions()[-1])

        # Store some information
        record(
            rebuilding_status=1,
            brick_size=context.last_brick_size,
            price=history[-1],
            renko_price=context.model.get_renko_prices()[-1],
            num_created_bars=0,
            amount=context.portfolio.positions[context.asset].amount
        )

    else:
        last_price = data.history(context.asset,
                                  'price',
                                  bar_count=1,
                                  frequency='1440T',
                                  )

        # Just for output and debug
        prev = context.model.get_renko_prices()[-1]
        prev_dir = context.model.get_renko_directions()[-1]
        num_created_bars = context.model.do_next(last_price)
        if num_created_bars != 0:
            print('New Renko bars created')
            print('last price: ' + str(last_price))
            print('previous Renko price: ' + str(prev))
            print('current Renko price: ' + str(context.model.get_renko_prices()[-1]))
            print('direction: ' + str(prev_dir))
            print('brick size: ' + str(context.model.brick_size))

        # Store some information
        record(
            rebuilding_status=0,
            brick_size=context.last_brick_size,
            price=last_price,
            renko_price=context.model.get_renko_prices()[-1],
            num_created_bars=num_created_bars,
            amount=context.portfolio.positions[context.asset].amount
        )

        # If the last price moves in the backward direction we should rebuild the model
        if np.sign(context.portfolio.positions[context.asset].amount * context.model.get_renko_directions()[-1]) == -1:
            order_target_percent(context.asset, 0.0)
            context.model = renko()
        # or we cover the part of the position
        elif context.part_cover_ratio > 0.0 and num_created_bars != 0:
            order_target(context.asset,
                         context.portfolio.positions[context.asset].amount * (1.0 - context.part_cover_ratio))


def analyze(context, perf):
    # Summary output

    print('Total return: ' + str(perf.algorithm_period_return[-1]*100))
    print('Sortino coef: ' + str(perf.sortino[-1]))
    print('Max drawdown: ' + str(np.min(perf.max_drawdown)))
    print('Alpha: ' + str(perf.alpha[-1]))
    print('Beta: ' + str(perf.beta[-1]))
    perf.to_csv('perf_' + str(context.asset) + '.csv')

    f = plt.figure(figsize=(7.2, 7.2))

    # Plot performance
    ax1 = f.add_subplot(611)
    ax1.plot(perf.algorithm_period_return, 'blue')
    ax1.plot(perf.benchmark_period_return, 'red')
    ax1.set_title('Performance')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Return')

    # Plot price and renko price
    ax2 = f.add_subplot(612, sharex=ax1)
    ax2.plot(perf.price, 'grey')
    ax2.plot(perf.renko_price, 'yellow')
    ax2.set_title(context.asset)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')

    transaction_df = extract_transactions(perf)
    if not transaction_df.empty:
        try:
            transactions = transaction_df['sid'] == context.asset
            transactions = transaction_df.loc[transactions]
        except IndexError:
            log.info(f"SID: \n {IndexError} ")
        try:
            buy_df = transactions.loc[transactions['amount'] > 0]
            sell_df = transactions.loc[transactions['amount'] < 0]

            # temp hardcoded plotting should be based on # of assets
            ax2.scatter(
                buy_df.index.to_pydatetime(),
                perf.loc[buy_df.index, 'price'],
                marker='^',
                s=100,
                c='green',
                label=''
            )
            ax2.scatter(
                sell_df.index.to_pydatetime(),
                perf.loc[sell_df.index, 'price'],
                marker='v',
                s=100,
                c='red',
                label=''
            )
        except (KeyError, UnboundLocalError) as e:
            log.info(f"SID: \n {e} ")

    # Plot brick size
    ax3 = f.add_subplot(613, sharex=ax1)
    ax3.plot(perf.brick_size, 'blue')
    xcoords = perf.index[perf.rebuilding_status == 1]
    for xc in xcoords:
        ax3.axvline(x=xc, color='red')
    ax3.set_title('Brick size and rebuilding status')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Size and Status')

    # Plot renko_price
    ax4 = f.add_subplot(614, sharex=ax1)
    ax4.plot(perf.num_created_bars, 'green')
    ax4.set_title('Number of created Renko bars')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Amount')

    # Plot amount of asset in portfolio
    ax5 = f.add_subplot(615, sharex=ax1)
    ax5.plot(perf.amount, 'black')
    ax5.set_title('Asset amount in portfolio')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Amount')

    # Plot drawdown
    ax6 = f.add_subplot(616, sharex=ax1)
    ax6.plot(perf.max_drawdown, 'yellow')
    ax6.set_title('Max drawdown')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Drawdown')
    plt.show()


if __name__ == '__main__':
    run_algorithm(
        capital_base=100000,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='binance',
        quote_currency='usdt',
        start=pd.to_datetime('2018-6-1', utc=True),
        end=pd.to_datetime('2019-2-10', utc=True)
    )
