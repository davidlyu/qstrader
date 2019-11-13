from .base import AbstractStatistics
from ..price_parser import PriceParser

from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from datetime import datetime

import qstrader.statistics.performance as perf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
import os

import pyecharts.options as opts
from pyecharts.charts import Line, Page, HeatMap, Bar
from pyecharts.components import Table

def _time_to_str(timestamp):
    return "{d.year}/{d.month}/{d.day}".format(d=timestamp)


class TearsheetStatistics(AbstractStatistics):
    """
    Displays a Matplotlib-generated 'one-pager' as often
    found in institutional strategy performance reports.

    Includes an equity curve, drawdown curve, monthly
    returns heatmap, yearly returns summary, strategy-
    level statistics and trade-level statistics.

    Also includes an optional annualised rolling Sharpe
    ratio chart.
    """
    def __init__(
        self, config, portfolio_handler,
        title=None, benchmark=None, periods=252,
        rolling_sharpe=False
    ):
        """
        Takes in a portfolio handler.
        """
        self.config = config
        self.portfolio_handler = portfolio_handler
        self.price_handler = portfolio_handler.price_handler
        self.title = '\n'.join(title)
        self.benchmark = benchmark
        self.periods = periods
        self.rolling_sharpe = rolling_sharpe
        self.equity = {}
        self.equity_benchmark = {}
        self.log_scale = False

    def update(self, timestamp, portfolio_handler):
        """
        Update equity curve and benchmark equity curve that must be tracked
        over time.
        """
        self.equity[timestamp] = PriceParser.display(
            self.portfolio_handler.portfolio.equity
        )
        if self.benchmark is not None:
            self.equity_benchmark[timestamp] = PriceParser.display(
                self.price_handler.get_last_close(self.benchmark)
            )

    def get_results(self):
        """
        Return a dict with all important results & stats.
        """
        # Equity
        equity_s = pd.Series(self.equity).sort_index()

        # Returns
        returns_s = equity_s.pct_change().fillna(0.0)

        # Rolling Annualised Sharpe
        rolling = returns_s.rolling(window=self.periods)
        rolling_sharpe_s = np.sqrt(self.periods) * (
            rolling.mean() / rolling.std()
        )

        # Cummulative Returns
        cum_returns_s = np.exp(np.log(1 + returns_s).cumsum())

        # Drawdown, max drawdown, max drawdown duration
        dd_s, max_dd, dd_dur = perf.create_drawdowns(cum_returns_s)

        statistics = {}

        # Equity statistics
        statistics["sharpe"] = perf.create_sharpe_ratio(
            returns_s, self.periods
        )
        statistics["drawdowns"] = dd_s
        # TODO: need to have max_drawdown so it can be printed at end of test
        statistics["max_drawdown"] = max_dd
        statistics["max_drawdown_pct"] = max_dd
        statistics["max_drawdown_duration"] = dd_dur
        statistics["equity"] = equity_s
        statistics["returns"] = returns_s
        statistics["rolling_sharpe"] = rolling_sharpe_s
        statistics["cum_returns"] = cum_returns_s

        positions = self._get_positions()
        if positions is not None:
            statistics["positions"] = positions

        # Benchmark statistics if benchmark ticker specified
        if self.benchmark is not None:
            equity_b = pd.Series(self.equity_benchmark).sort_index()
            returns_b = equity_b.pct_change().fillna(0.0)
            rolling_b = returns_b.rolling(window=self.periods)
            rolling_sharpe_b = np.sqrt(self.periods) * (
                rolling_b.mean() / rolling_b.std()
            )
            cum_returns_b = np.exp(np.log(1 + returns_b).cumsum())
            dd_b, max_dd_b, dd_dur_b = perf.create_drawdowns(cum_returns_b)
            statistics["sharpe_b"] = perf.create_sharpe_ratio(returns_b)
            statistics["drawdowns_b"] = dd_b
            statistics["max_drawdown_pct_b"] = max_dd_b
            statistics["max_drawdown_duration_b"] = dd_dur_b
            statistics["equity_b"] = equity_b
            statistics["returns_b"] = returns_b
            statistics["rolling_sharpe_b"] = rolling_sharpe_b
            statistics["cum_returns_b"] = cum_returns_b

        return statistics

    def _get_positions(self):
        """
        Retrieve the list of closed Positions objects from the portfolio
        and reformat into a pandas dataframe to be returned
        """
        def x(p):
            return PriceParser.display(p)

        pos = self.portfolio_handler.portfolio.closed_positions
        a = []
        for p in pos:
            a.append(p.__dict__)
        if len(a) == 0:
            # There are no closed positions
            return None
        else:
            df = pd.DataFrame(a)
            df['avg_bot'] = df['avg_bot'].apply(x)
            df['avg_price'] = df['avg_price'].apply(x)
            df['avg_sld'] = df['avg_sld'].apply(x)
            df['cost_basis'] = df['cost_basis'].apply(x)
            df['init_commission'] = df['init_commission'].apply(x)
            df['init_price'] = df['init_price'].apply(x)
            df['market_value'] = df['market_value'].apply(x)
            df['net'] = df['net'].apply(x)
            df['net_incl_comm'] = df['net_incl_comm'].apply(x)
            df['net_total'] = df['net_total'].apply(x)
            df['realised_pnl'] = df['realised_pnl'].apply(x)
            df['total_bot'] = df['total_bot'].apply(x)
            df['total_commission'] = df['total_commission'].apply(x)
            df['total_sld'] = df['total_sld'].apply(x)
            df['unrealised_pnl'] = df['unrealised_pnl'].apply(x)
            df['trade_pct'] = (df['avg_sld'] / df['avg_bot'] - 1.0)
            return df

    def _plot_equity(self, stats):
        """
        Plots cumulative rolling returns versus some benchmark.
        """
        def format_two_dec(x, pos):
            return '%.2f' % x

        equity = stats['cum_returns']
        list(equity.index.map(_time_to_str))

        c = (
            Line()
            .add_xaxis(list(equity.index.map(_time_to_str)))
            .add_yaxis("equity", list(equity), is_symbol_show=False)
            # .extend_axis(
            #     xaxis=opts.AxisOpts(
            #         axisline_opts=opts.AxisLineOpts(
            #             is_on_zero=False
            #         )
            #     )
            # )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Returns"),
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(is_scale=True)
                )
        )

        if self.benchmark is not None:
            benchmark = stats['cum_returns_b']
            c.add_yaxis("benchmark", list(benchmark), is_symbol_show=False)
        
        return c


    def _plot_drawdown(self, stats, ax=None, **kwargs):
        """
        Plots the underwater curve
        """
        def format_perc(x, pos):
            return '%.0f%%' % x

        drawdown = stats['drawdowns']
        c = (
            Line()
            .add_xaxis(list(drawdown.index.map(_time_to_str)))
            .add_yaxis("drawdown", list(drawdown), is_symbol_show=False)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Drawdown (%)"))
        )
        return c

    
    def _plot_monthly_returns(self, stats):
        """
        Plots the monthly returns heatmap
        """
        returns = stats['returns']
        monthly_ret = perf.aggregate_returns(returns, 'monthly')
        monthly_ret = monthly_ret.unstack()
        monthly_ret = np.round(monthly_ret, 3)
        monthly_ret = monthly_ret.fillna(0) * 100
        
        row_count = len(monthly_ret)
        month = [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ]
        values = []
        for i in range(12):
            for j in range(row_count):
                v = monthly_ret.iloc[j, i]
                s = "%.3f" % v
                values.append([i, j, round(v, 3), s])
        c = (
            HeatMap()
            .add_xaxis(month)
            .add_yaxis("", list(monthly_ret.index), values)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Monthly Returns (%)"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=min(values, key=lambda x: x[2])[2],
                    max_=max(values, key=lambda x: x[2])[2]),
            )
        )
        return c


    def _plot_yearly_returns(self, stats):
        """
        Plot the barplot of returns by year
        """
        returns = stats["returns"]
        yly_ret = perf.aggregate_returns(returns, 'yearly') * 100.0
        yly_ret = yly_ret.map(lambda x: round(x, 3))
        c = (
            Bar()
            .add_xaxis(list(yly_ret.index))
            .add_yaxis("", list(yly_ret))
            .set_global_opts(title_opts=opts.TitleOpts(
                title="Yearly Returns (%)"
            ))
        )
        return c


    def _plot_txt_curve(self, stats):
        """
        Output the statistics for the equity curve
        """
        returns = stats["returns"]
        cum_returns = stats['cum_returns']

        if 'positions' not in stats:
            trd_yr = 0
        else:
            positions = stats['positions']
            trd_yr = positions.shape[0] / (
                (returns.index[-1] - returns.index[0]).days / 365.0
            )

        tot_ret = cum_returns[-1] - 1.0
        cagr = perf.create_cagr(cum_returns, self.periods)
        sharpe = perf.create_sharpe_ratio(returns, self.periods)
        sortino = perf.create_sortino_ratio(returns, self.periods)
        rsq = perf.rsquared(range(cum_returns.shape[0]), cum_returns)
        dd, dd_max, dd_dur = perf.create_drawdowns(cum_returns)

        header = [
            "Performance", "Value"
        ]
        rows = [
            ["Total Return", "{:.0%}".format(tot_ret)],
            ["CAGR", "{:.2%}".format(cagr)],
            ["Sharpe Ratio", "{:.2f}".format(sharpe)],
            ["Sortino Ratio", "{:.2f}".format(sortino)],
            ["Annual Volatility", "{:.2%}".format(returns.std() * np.sqrt(252))],
            ["R-Squared", '{:.2f}'.format(rsq)],
            ["Max Daily Drawdown", '{:.2%}'.format(dd_max)],
            ["Max Drawdown Duration", '{:.0f}'.format(dd_dur)],
            ["Trades per Year", '{:.1f}'.format(trd_yr)]
        ]

        table = (
            Table()
            .add(header, rows)
            .set_global_opts(
                title_opts=opts.ComponentTitleOpts(
                    title="Curve"
                )
            )
        )
        return table

    def _plot_txt_trade(self, stats):
        """
        Outputs the statistics for the trades.
        """
        if 'positions' not in stats:
            num_trades = 0
            win_pct = "N/A"
            win_pct_str = "N/A"
            avg_trd_pct = "N/A"
            avg_win_pct = "N/A"
            avg_loss_pct = "N/A"
            max_win_pct = "N/A"
            max_loss_pct = "N/A"
        else:
            pos = stats['positions']
            num_trades = pos.shape[0]
            win_pct = pos[pos["trade_pct"] > 0].shape[0] / float(num_trades)
            win_pct_str = '{:.0%}'.format(win_pct)
            avg_trd_pct = '{:.2%}'.format(np.mean(pos["trade_pct"]))
            avg_win_pct = '{:.2%}'.format(np.mean(pos[pos["trade_pct"] > 0]["trade_pct"]))
            avg_loss_pct = '{:.2%}'.format(np.mean(pos[pos["trade_pct"] <= 0]["trade_pct"]))
            max_win_pct = '{:.2%}'.format(np.max(pos["trade_pct"]))
            max_loss_pct = '{:.2%}'.format(np.min(pos["trade_pct"]))

        max_loss_dt = 'TBD'  # pos[pos["trade_pct"] == np.min(pos["trade_pct"])].entry_date.values[0]
        avg_dit = '0.0'  # = '{:.2f}'.format(np.mean(pos.time_in_pos))

        header = [
            "Performance", "Value"
        ]
        rows = [
            ["Trade Winning %", "{}".format(win_pct_str)],
            ["Average Trade %", "{}".format(avg_trd_pct)],
            ["Average Win", "{}".format(avg_win_pct)],
            ["Average Loss", "{}".format(avg_loss_pct)],
            ["Best Trade", "{}".format(max_win_pct)],
            ["Worst Trade", '{}'.format(max_loss_pct)],
            ["Worst Trade Date", '{}'.format(max_loss_dt)],
            ["Avg Days in Trade", '{}'.format(avg_dit)],
            ["Trades", '{:.1f}'.format(num_trades)]
        ]

        table = (
            Table()
            .add(header, rows)
            .set_global_opts(
                title_opts=opts.ComponentTitleOpts(
                    title="Trade"
                )
            )
        )
        return table

    def _plot_txt_time(self, stats, ax=None, **kwargs):
        """
        Outputs the statistics for various time frames.
        """
        returns = stats['returns']

        mly_ret = perf.aggregate_returns(returns, 'monthly')
        yly_ret = perf.aggregate_returns(returns, 'yearly')

        mly_pct = mly_ret[mly_ret >= 0].shape[0] / float(mly_ret.shape[0])
        mly_avg_win_pct = np.mean(mly_ret[mly_ret >= 0])
        mly_avg_loss_pct = np.mean(mly_ret[mly_ret < 0])
        mly_max_win_pct = np.max(mly_ret)
        mly_max_loss_pct = np.min(mly_ret)
        yly_pct = yly_ret[yly_ret >= 0].shape[0] / float(yly_ret.shape[0])
        yly_max_win_pct = np.max(yly_ret)
        yly_max_loss_pct = np.min(yly_ret)

        header = [
            "Performance", "Value"
        ]
        rows = [
            ["Winning Months %", "{:.0%}".format(mly_pct)],
            ["Average Winning Month %", "{:.2%}".format(mly_avg_win_pct)],
            ["Average Losing Month %", "{:.2%}".format(mly_avg_loss_pct)],
            ["Best Month %", "{:.2%}".format(mly_max_win_pct)],
            ["Worst Month %", "{:.2%}".format(mly_max_loss_pct)],
            ["Winning Years %", '{:.0%}'.format(yly_pct)],
            ["Best Year %", '{:.2%}'.format(yly_max_win_pct)],
            ["Worst Year %", '{:.2%}'.format(yly_max_loss_pct)]
        ]

        table = (
            Table()
            .add(header, rows)
            .set_global_opts(
                title_opts=opts.ComponentTitleOpts(
                    title="Time"
                )
            )
        )
        return table


    def plot_results(self, filename=None):
        """
        Plot the Tearsheet
        """
        stats = self.get_results()
        output_dir = os.path.expanduser(self.config.OUTPUT_DIR)
        filename = os.path.join(output_dir, "render.html")

        self._plot_equity(stats).render(filename)
        equity_plot = self._plot_equity(stats)
        drawdown_plot = self._plot_drawdown(stats)
        monthly_returns_plot = self._plot_monthly_returns(stats)
        yearly_returns_plot = self._plot_yearly_returns(stats)
        curve_table = self._plot_txt_curve(stats)
        trade_table = self._plot_txt_trade(stats)
        time_table = self._plot_txt_time(stats)
        c = (
            Page()
            .add(equity_plot)
            .add(drawdown_plot)
            .add(monthly_returns_plot)
            .add(yearly_returns_plot)
            .add(curve_table)
            .add(trade_table)
            .add(time_table)
        )
        c.render(filename)


    def get_filename(self, filename=""):
        if filename == "":
            now = datetime.utcnow()
            filename = "tearsheet_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            filename = os.path.expanduser(os.path.join(self.config.OUTPUT_DIR, filename))
        return filename

    def save(self, filename=""):
        filename = self.get_filename(filename)
        self.plot_results
