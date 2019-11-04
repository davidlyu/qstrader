import os

import pandas as pd
import MySQLdb as mdb

from ..price_parser import PriceParser
from .base import AbstractBarPriceHandler
from ..event import BarEvent


DEFAULT_DB_HOST = "192.168.31.30"
DEFAULT_DB_USER = "sec_user"
DEFAULT_DB_PASS = "password"
DEFAULT_DB_NAME = "securities_master"


class DailySqlBarPriceHandler(AbstractBarPriceHandler):
    """
    YahooDailyBarPriceHandler is designed to read CSV files of
    Yahoo Finance daily Open-High-Low-Close-Volume (OHLCV) data
    for each requested financial instrument and stream those to
    the provided events queue as BarEvents.
    """
    def __init__(
        self, events_queue,
        db_host=None, db_user=None,
        db_pass=None, db_name=None,
        init_tickers=None,
        start_date=None, end_date=None,
        calc_adj_returns=False
    ):
        """
        Takes the CSV directory, the events queue and a possible
        list of initial ticker symbols then creates an (optional)
        list of ticker subscriptions and associated prices.
        """
        self.db_host = DEFAULT_DB_HOST if db_host is None else db_host
        self.db_user = DEFAULT_DB_USER if db_user is None else db_user
        self.db_pass = DEFAULT_DB_PASS if db_pass is None else db_pass
        self.db_name = DEFAULT_DB_NAME if db_name is None else db_name
        self.events_queue = events_queue
        self.continue_backtest = True
        self.tickers = {}
        self.tickers_data = {}
        self.start_date = start_date
        self.end_date = end_date
        if init_tickers is not None:
            for ticker in init_tickers:
                self.subscribe_ticker(ticker)
        self.bar_stream = self._merge_sort_ticker_data()
        self.calc_adj_returns = calc_adj_returns
        if self.calc_adj_returns:
            self.adj_close_returns = []


    def _price_read(self, ticker):
        """
        read price of equities from database,
        convert them into a pandas DataFrame, 
        and return them.
        """
        con = mdb.connect(
            self.db_host, self.db_user,
            self.db_pass, self.db_name
        )
        str_start = self.start_date.strftime("%Y-%m-%d")
        str_end = self.end_date.strftime("%Y-%m-%d")
        sql = f"""
        SELECT dp.price_date, dp.high_price, dp.low_price,
          dp.open_price, dp.adj_close_price, dp.close_price, dp.volume
        FROM symbol AS sym
        INNER JOIN daily_price AS dp
          ON dp.symbol_id = sym.id
        WHERE sym.ticker = '{ticker}'
          AND dp.price_date >= '{str_start}'
          AND dp.price_date <= '{str_end}'
        ORDER BY dp.price_date ASC;
        """
        df = pd.read_sql_query(sql, con=con)
        columns_name_map = {
            "price_date": "Date", "high_price": "High",
            "low_price": "Low", "open_price": "Open",
            "close_price": "Close", "adj_close_price": "Adj_Close",
            "volume": "Volume"
        }
        df.rename(columns_name_map, axis="columns", inplace=True)
        df.set_index("Date", inplace=True)
        return df


    def _open_ticker_price_sql(self, ticker):
        """
        Opens the database containing the equities ticks from
        the specified mysql database, converting them into
        them into a pandas DataFrame, stored in a dictionary.
        """
        df = self._price_read(ticker)
        self.tickers_data[ticker] = df
        self.tickers_data[ticker]["Ticker"] = ticker


    def _merge_sort_ticker_data(self):
        """
        Concatenates all of the separate equities DataFrames
        into a single DataFrame that is time ordered, allowing tick
        data events to be added to the queue in a chronological fashion.

        Note that this is an idealised situation, utilised solely for
        backtesting. In live trading ticks may arrive "out of order".
        """
        # reindex all ticker DataFrame, so that all DataFrame has same lenght.
        idx = None
        for df in self.tickers_data.values():
            if idx is None:
                idx = df.index
            else:
                idx = idx.union(df.index)
        for ticker in self.tickers_data:
            self.tickers_data[ticker] = self.tickers_data[ticker].reindex(index=idx, method="ffill")

        df = pd.concat(self.tickers_data.values()).sort_index()
        start = None
        end = None
        if self.start_date is not None:
            start = df.index.searchsorted(self.start_date)
        if self.end_date is not None:
            end = df.index.searchsorted(self.end_date)
        # This is added so that the ticker events are
        # always deterministic, otherwise unit test values
        # will differ
        df['colFromIndex'] = df.index
        df = df.sort_values(by=["colFromIndex", "Ticker"])
        if start is None and end is None:
            return df.iterrows()
        elif start is not None and end is None:
            return df.iloc[start:].iterrows()
        elif start is None and end is not None:
            return df.iloc[:end].iterrows()
        else:
            return df.iloc[start:end].iterrows()

    def subscribe_ticker(self, ticker):
        """
        Subscribes the price handler to a new ticker symbol.
        """
        if ticker not in self.tickers:
            try:
                self._open_ticker_price_sql(ticker)
                dft = self.tickers_data[ticker]
                row0 = dft.iloc[0]

                close = PriceParser.parse(row0["Close"])
                adj_close = PriceParser.parse(row0["Adj_Close"])

                ticker_prices = {
                    "close": close,
                    "adj_close": adj_close,
                    "timestamp": dft.index[0]
                }
                self.tickers[ticker] = ticker_prices
            except OSError:
                print(
                    "Could not subscribe ticker %s "
                    "as no data CSV found for pricing." % ticker
                )
        else:
            print(
                "Could not subscribe ticker %s "
                "as is already subscribed." % ticker
            )

    def _create_event(self, index, period, ticker, row):
        """
        Obtain all elements of the bar from a row of dataframe
        and return a BarEvent
        """
        open_price = PriceParser.parse(row["Open"])
        high_price = PriceParser.parse(row["High"])
        low_price = PriceParser.parse(row["Low"])
        close_price = PriceParser.parse(row["Close"])
        adj_close_price = PriceParser.parse(row["Adj_Close"])
        volume = int(row["Volume"])
        bev = BarEvent(
            ticker, index, period, open_price,
            high_price, low_price, close_price,
            volume, adj_close_price
        )
        return bev

    def _store_event(self, event):
        """
        Store price event for closing price and adjusted closing price
        """
        ticker = event.ticker
        # If the calc_adj_returns flag is True, then calculate
        # and store the full list of adjusted closing price
        # percentage returns in a list
        # TODO: Make this faster
        if self.calc_adj_returns:
            prev_adj_close = self.tickers[ticker][
                "adj_close"
            ] / float(PriceParser.PRICE_MULTIPLIER)
            cur_adj_close = event.adj_close_price / float(
                PriceParser.PRICE_MULTIPLIER
            )
            self.tickers[ticker][
                "adj_close_ret"
            ] = cur_adj_close / prev_adj_close - 1.0
            self.adj_close_returns.append(self.tickers[ticker]["adj_close_ret"])
        self.tickers[ticker]["close"] = event.close_price
        self.tickers[ticker]["adj_close"] = event.adj_close_price
        self.tickers[ticker]["timestamp"] = event.time

    def stream_next(self):
        """
        Place the next BarEvent onto the event queue.
        """
        try:
            index, row = next(self.bar_stream)
        except StopIteration:
            self.continue_backtest = False
            return
        # Obtain all elements of the bar from the dataframe
        ticker = row["Ticker"]
        period = 86400  # Seconds in a day
        # Create the tick event for the queue
        bev = self._create_event(index, period, ticker, row)
        # Store event
        self._store_event(bev)
        # Send event to queue
        self.events_queue.put(bev)