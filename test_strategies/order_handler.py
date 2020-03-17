#The order handler is created in the initalize method in main execution file



class OrderHandler(object):
    """
    The Orderhandler class receives positions sizes (defines long or short dichotomy), time execution (limit to market type)
    , and the asset to trade from strategies,. Orderhandler and strategies are connected through a global list containing
    order objects (or an ordered dictionary)

    Order object / Dict:
    - Asset. symbol
    - Execution time (int var)
    - Id or reference to order creator (no double spend or buy):
        Could be fixed by creating a string object for each crypto_asset class then adding that
        as a unique identifier and a
    - order_id (returned by ordertarget percent)

    """
    def __init__(self, **kwargs):
        """
        :param kwargs:
        - Contains reference to orderlist
        :return:
        """

    def limit_orders_within_price_range(self):
        """
        Checks if the orders limit price is at or price the current bid / ask. (or within set parametrs (say 1-10 dollars)
        """
    def orders_uptick(self):
        """
        :param orderlist: up the tickers in all orders
        :return: (void function doesnt execute
        """
    def re_limit_order(self, order):
        """
        cancells and creates a new limit order in the order object
        :param order:
        :return:
        """

    def market(self, order):
        """

        :param order:
        :return:
        """

    def new_order(self, **kwargs):

