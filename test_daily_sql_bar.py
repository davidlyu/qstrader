from queue import Queue

from qstrader.price_handler.daily_sql_bar import DailySqlBarPriceHandler


def _test_price_reader():
    event_queue = Queue()
    han = DailySqlBarPriceHandler(events_queue=event_queue, init_tickers=["CSI300", "CSI500"])
    df = han._price_read("CSI300")
    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    _test_price_reader()