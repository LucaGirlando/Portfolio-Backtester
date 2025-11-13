import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import skew as stat_skew, kurtosis as stat_kurtosis
from datetime import datetime
import io

# Configuration
st.set_page_config(
    page_title="Portfolio Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

TRADING_DAYS = 252

# Metric documentation and explanation
METRIC_DOCS = {
    'Total Return': "Total Return = (Final Value / Initial Capital) - 1.\n\nAggregate gain or loss over the period.",
    'CAGR': "CAGR = (1 + Total Return)^(252 / Ndays) - 1.\n\nCAGR stands for Compound Annual Growth Rate.\nThe steady annual growth rate that would lead to the same final value.",
    'Volatility (ann)': "Annualized Volatility = stdev(daily returns) * sqrt(252).\n\nDispersion of returns.\nIs a statistical measure of the returns for a given security or market index over time.\nIt is often measured from either the standard deviation or variance between those returns. In most cases, the higher the volatility, the riskier the security.",
    'Sharpe (ann)': "Sharpe = (Annualized Return - Risk-Free Rate) / Annualized Volatility.\n\nExcess return per unit of risk.\nThe Sharpe ratio shows whether a portfolio's excess returns are attributable to smart investment decisions or luck and risk.\nA measure of an investment's risk-adjusted performance, calculated by comparing its return to that of a risk-free asset",
    'Sortino (ann)': "Sortino = (Annualized Return - Risk-Free Rate) / Downside Volatility.\n\nA risk-adjusted measure of portfolio performance that only considers the standard deviation of the downside risk.\nThe Sortino ratio can help investors and analysts evaluate an investment's return for a degree of bad risk.\nPenalizes downside only.",
    'Max Drawdown': "Max Drawdown = min(Equity / Rolling Max - 1).\n\nWorst peak-to-trough decline.",
    'Calmar': "Calmar = CAGR / |Max Drawdown|.\n\nIt is a function of the fund's average compounded annual rate of return versus its maximum drawdown. The higher the Calmar ratio, the better it performed on a risk-adjusted basis during the given time frame, which is mostly commonly set at 36 months.\nGrowth relative to drawdowns.",
    'Daily VaR 95%': "Historical VaR(95%) = 5th percentile of daily returns.\n\nLoss threshold exceeded 5% of the time.",
    'Daily CVaR 95%': "Historical CVaR(95%) = average of daily returns at or below the 95% VaR (expected shortfall).",
    'Daily VaR 99%': "Historical VaR(99%) = 1st percentile of daily returns.",
    'Daily CVaR 99%': "Historical CVaR(99%) = average of daily returns at or below the 99% VaR.",
    'Hit Ratio': "Hit Ratio = fraction of days with positive returns.",
    'Best Day': "Best Day = maximum daily return.",
    'Worst Day': "Worst Day = minimum daily return.",
    'Skew': "Skewness of daily returns.\n\nSkewness is a measure of symmetry, or more precisely, the lack of symmetry. A distribution, or data set, is symmetric if it looks the same to the left and right of the center point.\nPositive skew: fatter right tail; negative: fatter left tail.",
    'Kurtosis (excess)': "Excess kurtosis of daily returns relative to normal distribution.\nKurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. A uniform distribution would be the extreme case.",
    'Beta vs Benchmark': "Beta = Cov(Rp,Rb)/Var(Rb).\n\nRp and Rb are the portfolio returns and the benchmark returns\nSensitivity to the benchmark.",
    'Alpha (ann) vs Benchmark': "Alpha = Annualized Return - [Rf + Beta*(Annualized Benchmark Return - Rf)].\n\nAlpha iss a term used in investing to describe an investment strategy's ability to beat the market.\nIs often referred to as excess return or the abnormal rate of return in relation to the benchmark.",
    'Correlation vs Benchmark': "Correlation of daily returns with the benchmark.",
    'R^2 vs Benchmark': "R-squared = Correlation^2.\n\nVariance explained by the benchmark.",
    'Initial Capital': "Starting portfolio value used for the backtest.",
    'Final Value': "Ending portfolio value over the backtest horizon.",
}

def parse_float_list(s):
    if not s.strip():
        return None
    try:
        vals = [float(x.strip()) for x in s.split(',') if x.strip() != ""]
        return vals
    except Exception:
        return None

def parse_ticker_list(s):
    return [t.strip().upper() for t in s.split(',') if t.strip()]

def annualize_return(daily_mean):
    return daily_mean * TRADING_DAYS

def annualize_vol(daily_std):
    return daily_std * np.sqrt(TRADING_DAYS)

def compute_drawdown(value_series: pd.Series):
    roll_max = value_series.cummax()
    dd = value_series / roll_max - 1.0
    max_dd = dd.min()
    max_dd_duration = (dd < 0).astype(int).groupby((dd >= 0).astype(int).cumsum()).cumcount().max()
    return dd, max_dd, max_dd_duration

def monthly_return_table(value_series: pd.Series):
    m = value_series.resample('M').last().pct_change().dropna()
    df = m.to_frame('Return')
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df['Month'] = df['Month'].map(month_names)
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    pivot = pivot.reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    return pivot

class PortfolioBacktester:
    def __init__(self, tickers, weights=None, start=None, end=None, initial_capital=10000.0,
                 rebalance='Monthly', rebalance_n=None, benchmark=None, risk_free=0.02,
                 tx_cost_bps=0.0, slippage_bps=0.0):
        self.tickers = tickers
        self.weights = None if weights is None else np.array(weights, dtype=float)
        self.start = start
        self.end = end
        self.initial_capital = float(initial_capital)
        self.rebalance = rebalance  # 'None', 'Monthly', 'Quarterly', 'Yearly', 'Every N days'
        self.rebalance_n = int(rebalance_n) if rebalance_n not in (None, "", "0") else None
        self.benchmark = benchmark
        self.risk_free = float(risk_free)
        self.tx_cost_rate = (float(tx_cost_bps) + float(slippage_bps)) / 10000.0
        
        # Weights or autoadjust to equal
        if self.weights is not None:
            if len(self.weights) != len(self.tickers):
                raise ValueError("Weights count must match number of tickers.")
            total = self.weights.sum()
            if total <= 0:
                raise ValueError("Weights must sum to > 0.")
            self.weights = self.weights / total  # normalize even if not exactly 1
        else:
            self.weights = np.ones(len(self.tickers)) / len(self.tickers)
    
    def _download_prices(self, tickers):
        data = yf.download(tickers, start=self.start, end=self.end, progress=False, auto_adjust=False)
        try:
            prices = data['Adj Close']
        except Exception:
            prices = data['Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.sort_index().ffill().dropna(how='any')
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)
        return prices
    
    def _rebalance_flags(self, index, freq):
        n = len(index)
        flags = np.zeros(n, dtype=bool)
        if n == 0:
            return flags
        flags[0] = True
        s = pd.Series(index=index, data=index)
        if not isinstance(index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
            s[:] = pd.to_datetime(s)
        if freq is None or freq == 'None':
            return flags
        if freq == 'Monthly':
            period = s.dt.to_period('M')
            changes = period != period.shift(1)
            return changes.fillna(True).values
        elif freq == 'Quarterly':
            period = s.dt.to_period('Q')
            changes = period != period.shift(1)
            return changes.fillna(True).values
        elif freq == 'Yearly':
            period = s.dt.to_period('Y')
            changes = period != period.shift(1)
            return changes.fillna(True).values
        elif freq == 'Every N days' and self.rebalance_n and self.rebalance_n > 0:
            idx = np.arange(n)
            flags[:] = (idx % self.rebalance_n == 0)
            return flags
        else:
            return flags

    def _compute_portfolio_path(self, prices: pd.DataFrame):
        idx = prices.index
        cols = list(prices.columns)
        W = self.weights.copy()
        rebal_flags = self._rebalance_flags(idx, self.rebalance)
        shares = pd.DataFrame(index=idx, columns=cols, dtype=float)
        tx_costs = pd.Series(0.0, index=idx, name='Tx Costs')
        turnover = pd.Series(0.0, index=idx, name='Turnover')

        # Initial allocation including costs
        initial_cost = self.initial_capital * self.tx_cost_rate
        capital_net = max(0.0, self.initial_capital - initial_cost)
        shares.iloc[0, :] = (capital_net * W) / prices.iloc[0, :].values
        tx_costs.iloc[0] = initial_cost
        turnover.iloc[0] = 1.0  # all capital used

        for i in range(1, len(idx)):
            prev_val = float((shares.iloc[i-1, :].values * prices.iloc[i, :].values).sum())
            if rebal_flags[i]:
                # Desired shares before cost
                target_shares_pre = (prev_val * W) / prices.iloc[i, :].values
                # Turnover value in $
                tv = np.abs(target_shares_pre - shares.iloc[i-1, :].values) * prices.iloc[i, :].values
                tv_sum = float(tv.sum())
                cost = tv_sum * self.tx_cost_rate
                net_val = max(0.0, prev_val - cost)
                shares.iloc[i, :] = (net_val * W) / prices.iloc[i, :].values
                tx_costs.iloc[i] = cost
                turnover.iloc[i] = tv_sum / prev_val if prev_val > 0 else 0.0
            else:
                shares.iloc[i, :] = shares.iloc[i-1, :].values
                tx_costs.iloc[i] = 0.0
                turnover.iloc[i] = 0.0

        # portfolio
        value = (shares * prices).sum(axis=1)
        weights_over_time = (shares * prices).div(value, axis=0)
        return value, shares, weights_over_time, tx_costs, turnover

    def run(self):
        prices = self._download_prices(self.tickers)
        if prices.empty:
            raise ValueError("No price data returned. Check tickers and dates.")
        port_value, shares, weights_ot, tx_costs, turnover = self._compute_portfolio_path(prices)
        port_rets = port_value.pct_change().dropna()

        # Benchmark data
        bench_prices = bench_rets = None
        if self.benchmark and self.benchmark.strip():
            bench_prices = self._download_prices([self.benchmark]).iloc[:, 0]
            bench_prices = bench_prices.reindex(port_value.index).dropna()
            common_idx = port_value.index.intersection(bench_prices.index)
            port_value = port_value.reindex(common_idx)
            port_rets = port_value.pct_change().dropna()
            bench_prices = bench_prices.reindex(common_idx)
            bench_rets = bench_prices.pct_change().dropna()
            tx_costs = tx_costs.reindex(common_idx).fillna(0.0)
            turnover = turnover.reindex(common_idx).fillna(0.0)

        metrics = self._compute_metrics(port_value, port_rets, bench_rets)
        contrib = self._compute_contributions(prices, shares, port_value.iloc[0], port_value.iloc[-1])
        asset_rets = prices.pct_change().dropna()
        corr = asset_rets.corr() if asset_rets.shape[1] > 1 else None
        monthly_tbl = monthly_return_table(port_value)

        result = {
            'prices': prices,
            'portfolio_value': port_value,
            'portfolio_returns': port_rets,
            'weights_over_time': weights_ot,
            'metrics': metrics,
            'contributions': contrib,
            'correlation': corr,
            'monthly_table': monthly_tbl,
            'benchmark_prices': bench_prices,
            'benchmark_returns': bench_rets,
            'tx_costs': tx_costs,
            'turnover': turnover,
        }
        return result

    def _compute_metrics(self, value: pd.Series, returns: pd.Series, bench_rets: pd.Series | None):
        rf = float(self.risk_free)
        total_return = value.iloc[-1] / value.iloc[0] - 1.0
        n_days = max(1, len(value) - 1)
        cagr = (1.0 + total_return) ** (TRADING_DAYS / n_days) - 1.0

        mu_daily = returns.mean()
        sd_daily = returns.std(ddof=0)
        mu_ann = annualize_return(mu_daily)
        vol_ann = annualize_vol(sd_daily)
        sharpe = (mu_ann - rf) / vol_ann if vol_ann != 0 else np.nan

        downside = returns[returns < 0]
        ds_std_daily = downside.std(ddof=0)
        ds_std_ann = annualize_vol(ds_std_daily)
        sortino = (mu_ann - rf) / ds_std_ann if ds_std_ann != 0 else np.nan

        _, max_dd, _ = compute_drawdown(value)
        calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan

        if len(returns) > 0:
            var95 = np.percentile(returns, 5)
            cvar95 = returns[returns <= var95].mean() if (returns <= var95).any() else np.nan
            var99 = np.percentile(returns, 1)
            cvar99 = returns[returns <= var99].mean() if (returns <= var99).any() else np.nan
        else:
            var95 = cvar95 = var99 = cvar99 = np.nan

        hit_ratio = (returns > 0).mean() if len(returns) > 0 else np.nan
        best_day = returns.max() if len(returns) > 0 else np.nan
        worst_day = returns.min() if len(returns) > 0 else np.nan

        skewv = stat_skew(returns, bias=False) if len(returns) > 2 else np.nan
        kurtv = stat_kurtosis(returns, fisher=True, bias=False) if len(returns) > 3 else np.nan

        beta = alpha_ann = r2 = corr = np.nan
        if bench_rets is not None and len(bench_rets) > 10:
            df = pd.DataFrame({'p': returns, 'b': bench_rets}).dropna()
            if len(df) > 10:
                cov = np.cov(df['p'], df['b'])[0, 1]
                var_b = np.var(df['b'])
                beta = cov / var_b if var_b != 0 else np.nan
                mu_b_ann = annualize_return(df['b'].mean())
                corr = df['p'].corr(df['b'])
                r2 = corr ** 2 if pd.notna(corr) else np.nan
                alpha_ann = mu_ann - (rf + beta * (mu_b_ann - rf)) if pd.notna(beta) else np.nan

        metrics = {
            'Start': value.index[0].strftime('%Y-%m-%d'),
            'End': value.index[-1].strftime('%Y-%m-%d'),
            'Initial Capital': value.iloc[0],
            'Final Value': value.iloc[-1],
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility (ann)': vol_ann,
            'Sharpe (ann)': sharpe,
            'Sortino (ann)': sortino,
            'Max Drawdown': max_dd,
            'Calmar': calmar,
            'Daily VaR 95%': var95,
            'Daily CVaR 95%': cvar95,
            'Daily VaR 99%': var99,
            'Daily CVaR 99%': cvar99,
            'Hit Ratio': hit_ratio,
            'Best Day': best_day,
            'Worst Day': worst_day,
            'Skew': skewv,
            'Kurtosis (excess)': kurtv,
        }
        if bench_rets is not None:
            metrics.update({
                'Beta vs Benchmark': beta,
                'Alpha (ann) vs Benchmark': alpha_ann,
                'Correlation vs Benchmark': corr,
                'R^2 vs Benchmark': r2,
            })
        return metrics

    def _compute_contributions(self, prices: pd.DataFrame, shares: pd.DataFrame, v0: float, vend: float):
        start_values = shares.iloc[0] * prices.iloc[0]
        end_values = shares.iloc[-1] * prices.iloc[-1]
        contrib = (end_values - start_values) / v0
        contrib.name = 'Contribution'
        contrib = contrib.sort_values(ascending=False)
        return contrib

def format_metric_value(key, val):
    if not isinstance(val, (int, float)) or pd.isna(val):
        return str(val)
    if key in ('Initial Capital', 'Final Value'):
        return f"${val:,.2f}"
    pct_keys = {'Total Return', 'CAGR', 'Volatility (ann)', 'Max Drawdown',
                'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%',
                'Hit Ratio', 'Best Day', 'Worst Day', 'Alpha (ann) vs Benchmark'}
    if key in pct_keys:
        return f"{val*100:.2f}%"
    if key in ('Sharpe (ann)', 'Sortino (ann)', 'Calmar', 'Beta vs Benchmark',
               'Correlation vs Benchmark', 'R^2 vs Benchmark', 'Skew', 'Kurtosis (excess)'):
        return f"{val:.2f}"
    return f"{val:.4f}"

def interpret_metrics(metrics):
    def pct(x):
        return f"{x*100:.2f}%" if pd.notna(x) else "n/a"
    def num(x):
        return f"{x:,.2f}" if pd.notna(x) else "n/a"
    
    lines = []
    try:
        init_v = float(metrics.get('Initial Capital', np.nan))
        end_v = float(metrics.get('Final Value', np.nan))
        lines.append(f"**Growth**: ${init_v:,.0f} → ${end_v:,.0f} ({pct(metrics.get('Total Return'))}) from {metrics.get('Start')} to {metrics.get('End')}.")
    except Exception:
        pass
    lines.append(f"**CAGR**: {pct(metrics.get('CAGR'))} | **Volatility (ann)**: {pct(metrics.get('Volatility (ann)'))}")
    lines.append(f"**Sharpe**: {num(metrics.get('Sharpe (ann)'))} | **Sortino**: {num(metrics.get('Sortino (ann)'))} | **Calmar**: {num(metrics.get('Calmar'))}")
    lines.append(f"**Max drawdown**: {pct(metrics.get('Max Drawdown'))}")
    lines.append(f"**Tail risk (daily)**: VaR95 {pct(metrics.get('Daily VaR 95%'))}, CVaR95 {pct(metrics.get('Daily CVaR 95%'))}")
    if 'Beta vs Benchmark' in metrics and pd.notna(metrics.get('Beta vs Benchmark')):
        lines.append(f"**Benchmark**: Beta {num(metrics.get('Beta vs Benchmark'))}, Alpha {pct(metrics.get('Alpha (ann) vs Benchmark'))}, Corr {num(metrics.get('Correlation vs Benchmark'))}, R² {num(metrics.get('R^2 vs Benchmark'))}")
    lines.append(f"**Hit ratio**: {pct(metrics.get('Hit Ratio'))} | **Best day**: {pct(metrics.get('Best Day'))} | **Worst day**: {pct(metrics.get('Worst Day'))}")
    lines.append(f"**Shape**: Skew {num(metrics.get('Skew'))}, Excess kurtosis {num(metrics.get('Kurtosis (excess)'))}")
    lines.append("\n**Notes**: Uses adjusted close (dividends included), fractional shares, optional fees/slippage, and scheduled rebalancing.")
    return "\n\n".join(lines)

def create_excel_export(result, mc_result=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Portfolio History
        pv = result['portfolio_value']
        pr = result['portfolio_returns']
        dd, _, _ = compute_drawdown(pv)
        
        hist_df = pd.concat([
            pv.rename('Portfolio Value'),
            pr.reindex(pv.index).rename('Portfolio Return'),
            dd.rename('Drawdown')
        ], axis=1)
        
        if result.get('benchmark_prices') is not None:
            hist_df = hist_df.join(result['benchmark_prices'].rename('Benchmark Price'))
            if result.get('benchmark_returns') is not None:
                hist_df = hist_df.join(result['benchmark_returns'].rename('Benchmark Return'))
        
        hist_df.index.name = 'Date'
        hist_df.reset_index(inplace=True)
        hist_df.to_excel(writer, sheet_name='Portfolio_History', index=False)
        
        # Metrics
        metrics_df = pd.DataFrame(list(result['metrics'].items()), columns=['Metric', 'Value'])
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Monte Carlo results if available
        if mc_result:
            mc_data = []
            for key, value in mc_result.items():
                if key not in ['q05', 'q50', 'q95', 'end_rets']:
                    mc_data.append([key, value])
            mc_df = pd.DataFrame(mc_data, columns=['Parameter', 'Value'])
            mc_df.to_excel(writer, sheet_name='Monte_Carlo', index=False)
    
    return output.getvalue()

# Streamlit App
def main():
    st.title("📈 Portfolio Backtester")
    st.markdown("""
    This application allows you to backtest portfolio strategies with various rebalancing options,
    calculate performance metrics, and run Monte Carlo simulations for risk analysis.
    
    **How to use:**
    1. Enter ticker symbols (comma-separated, e.g., AAPL,MSFT,GOOGL)
    2. Configure backtest parameters in the sidebar
    3. Click 'Run Backtest' to analyze performance
    4. View results in different tabs
    5. Run Monte Carlo simulations for forward-looking analysis
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Backtest Configuration")
        
        # Ticker input
        tickers = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL,MSFT,GOOGL",
            help="Enter stock tickers from Yahoo Finance (e.g., AAPL, MSFT, GOOGL)"
        )
        
        # Weights (optional)
        weights = st.text_input(
            "Weights (optional, comma-separated)",
            value="",
            help="Optional: Specify weights that sum to approximately 1 (e.g., 0.5,0.3,0.2). Leave empty for equal weighting."
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2018, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime.today())
        
        # Capital and risk-free rate
        initial_capital = st.number_input("Initial Capital ($)", value=100000.0, min_value=0.0)
        risk_free_rate = st.number_input("Risk-Free Rate (annual)", value=0.02, step=0.01)
        
        # Rebalancing options
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            options=["None", "Monthly", "Quarterly", "Yearly", "Every N days"],
            index=1
        )
        
        rebalance_n = None
        if rebalance_freq == "Every N days":
            rebalance_n = st.number_input("Rebalance Every N Days", value=30, min_value=1)
        
        # Benchmark
        benchmark = st.text_input(
            "Benchmark Ticker (optional)",
            value="^GSPC",
            help="Compare against a benchmark (e.g., ^GSPC for S&P 500)"
        )
        
        # Costs
        col3, col4 = st.columns(2)
        with col3:
            fee_bps = st.number_input("Fees (bps)", value=0.0, help="Transaction fees in basis points")
        with col4:
            slip_bps = st.number_input("Slippage (bps)", value=0.0, help="Slippage costs in basis points")
        
        # Log scale option
        log_scale = st.checkbox("Log Scale Equity Chart")
        
        # Run button
        if st.button("🚀 Run Backtest", use_container_width=True):
            st.session_state.run_backtest = True
    
    # Main content area
    if st.session_state.get('run_backtest', False):
        try:
            # Parse inputs
            ticker_list = parse_ticker_list(tickers)
            if not ticker_list:
                st.error("Please enter at least one valid ticker symbol.")
                return
                
            weight_list = parse_float_list(weights)
            
            # Run backtest
            with st.spinner("Running backtest... This may take a few moments."):
                bt = PortfolioBacktester(
                    tickers=ticker_list,
                    weights=weight_list,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    initial_capital=initial_capital,
                    rebalance=rebalance_freq,
                    rebalance_n=rebalance_n,
                    benchmark=benchmark if benchmark else None,
                    risk_free=risk_free_rate,
                    tx_cost_bps=fee_bps,
                    slippage_bps=slip_bps
                )
                result = bt.run()
                st.session_state.last_result = result
                st.success("Backtest completed successfully!")
        
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            return
    
    # Display results if available
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        metrics = result['metrics']
        
        # Summary metrics
        st.header("📊 Performance Summary")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", format_metric_value('Total Return', metrics['Total Return']))
            st.metric("CAGR", format_metric_value('CAGR', metrics['CAGR']))
        with col2:
            st.metric("Volatility (ann)", format_metric_value('Volatility (ann)', metrics['Volatility (ann)']))
            st.metric("Sharpe Ratio", format_metric_value('Sharpe (ann)', metrics['Sharpe (ann)']))
        with col3:
            st.metric("Max Drawdown", format_metric_value('Max Drawdown', metrics['Max Drawdown']))
            st.metric("Sortino Ratio", format_metric_value('Sortino (ann)', metrics['Sortino (ann)']))
        with col4:
            st.metric("Calmar Ratio", format_metric_value('Calmar', metrics['Calmar']))
            st.metric("Hit Ratio", format_metric_value('Hit Ratio', metrics['Hit Ratio']))
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Equity & Drawdown", 
            "📊 Metrics", 
            "🔄 Rolling Analysis", 
            "🔗 Correlations", 
            "📅 Monthly Returns", 
            "🎲 Monte Carlo"
        ])
        
        with tab1:
            # Equity curve and drawdown
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Equity curve
            pv = result['portfolio_value']
            bench = result['benchmark_prices']
            
            ax1.plot(pv.index, pv.values, label='Portfolio', linewidth=2)
            if bench is not None:
                bench_norm = bench / bench.iloc[0] * pv.iloc[0]
                ax1.plot(bench_norm.index, bench_norm.values, label='Benchmark (scaled)', alpha=0.8)
            ax1.set_title("Equity Curve")
            ax1.set_ylabel("Value ($)")
            ax1.legend()
            if log_scale:
                ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            dd, max_dd, _ = compute_drawdown(pv)
            ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
            ax2.set_title(f"Drawdown (Max: {max_dd*100:.2f}%)")
            ax2.set_ylabel("Drawdown")
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            **Equity Curve Interpretation:**
            - Shows the growth of your portfolio over time
            - Benchmark comparison helps assess relative performance
            - Log scale can help visualize percentage changes more clearly
            
            **Drawdown Interpretation:**
            - Measures peak-to-trough declines in portfolio value
            - Larger drawdowns indicate higher risk
            - Sustained drawdowns may indicate strategy issues
            """)
        
        with tab2:
            # Detailed metrics and contributions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                metrics_data = []
                order = [
                    'Start', 'End', 'Initial Capital', 'Final Value', 'Total Return', 'CAGR',
                    'Volatility (ann)', 'Sharpe (ann)', 'Sortino (ann)', 'Max Drawdown', 'Calmar',
                    'Daily VaR 95%', 'Daily CVaR 95%', 'Daily VaR 99%', 'Daily CVaR 99%',
                    'Hit Ratio', 'Best Day', 'Worst Day', 'Skew', 'Kurtosis (excess)'
                ]
                if 'Beta vs Benchmark' in metrics:
                    order.extend(['Beta vs Benchmark', 'Alpha (ann) vs Benchmark', 
                                'Correlation vs Benchmark', 'R^2 vs Benchmark'])
                
                for metric in order:
                    if metric in metrics:
                        metrics_data.append({
                            'Metric': metric,
                            'Value': format_metric_value(metric, metrics[metric])
                        })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Metric explanations
                selected_metric = st.selectbox("Select metric for explanation", list(METRIC_DOCS.keys()))
                st.info(f"**{selected_metric}**: {METRIC_DOCS[selected_metric]}")
            
            with col2:
                st.subheader("Asset Contributions")
                contrib_data = []
                for asset, contribution in result['contributions'].items():
                    contrib_data.append({
                        'Asset': asset,
                        'Contribution': f"{contribution*100:.2f}%"
                    })
                
                contrib_df = pd.DataFrame(contrib_data)
                st.dataframe(contrib_df, use_container_width=True)
                
                # Interpretation
                st.subheader("Performance Interpretation")
                st.markdown(interpret_metrics(metrics))
        
        with tab3:
            # Rolling analysis
            st.subheader("Rolling Risk Analysis")
            
            r = result['portfolio_returns']
            if len(r) > 20:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Rolling volatility
                win1 = 63
                roll_vol = r.rolling(win1).std() * np.sqrt(TRADING_DAYS)
                ax1.plot(roll_vol.index, roll_vol.values)
                ax1.set_title(f"Rolling Volatility ({win1}-day window)")
                ax1.set_ylabel("Annualized Volatility")
                ax1.grid(True, alpha=0.3)
                
                # Rolling Sharpe
                win2 = 252
                roll_mu_ann = r.rolling(win2).mean() * TRADING_DAYS
                roll_sd_ann = r.rolling(win2).std() * np.sqrt(TRADING_DAYS)
                roll_sharpe = (roll_mu_ann - risk_free_rate) / roll_sd_ann
                ax2.plot(roll_sharpe.index, roll_sharpe.values, color='green')
                ax2.set_title(f"Rolling Sharpe Ratio ({win2}-day window)")
                ax2.set_ylabel("Sharpe Ratio")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **Rolling Analysis Interpretation:**
                - **Volatility**: Shows how risk changes over time
                - **Sharpe Ratio**: Measures risk-adjusted performance over time
                - Consistent patterns may indicate regime changes or strategy effectiveness
                """)
            else:
                st.warning("Insufficient data for rolling analysis")
        
        with tab4:
            # Correlation heatmap
            st.subheader("Asset Correlation Matrix")
            
            corr = result['correlation']
            if corr is not None and corr.shape[0] > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, fmt=".2f", square=True)
                ax.set_title("Asset Return Correlations")
                st.pyplot(fig)
                
                st.markdown("""
                **Correlation Interpretation:**
                - Values close to +1: Assets move together
                - Values close to -1: Assets move oppositely  
                - Values near 0: No relationship
                - Lower correlations generally provide better diversification benefits
                """)
            else:
                st.info("Correlation heatmap requires 2 or more assets")
        
        with tab5:
            # Monthly returns heatmap
            st.subheader("Monthly Returns Heatmap")
            
            tbl = result['monthly_table']
            if tbl is not None and not tbl.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(tbl * 100.0, annot=True, fmt=".1f", cmap='RdYlGn', center=0, ax=ax)
                ax.set_title("Monthly Returns (%)")
                st.pyplot(fig)
                
                st.markdown("""
                **Monthly Returns Interpretation:**
                - Shows seasonal patterns in returns
                - Helps identify best/worst performing months
                - Consistent patterns may inform timing strategies
                """)
            else:
                st.warning("Insufficient data for monthly returns analysis")
        
        with tab6:
            # Monte Carlo simulation
            st.subheader("Monte Carlo Simulation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_paths = st.number_input("Number of Paths", value=1000, min_value=100, max_value=10000)
            with col2:
                years = st.number_input("Time Horizon (Years)", value=1.0, min_value=0.1, max_value=10.0)
            with col3:
                model = st.selectbox("Model", ["GBM (Normal)", "Bootstrap"])
            
            if st.button("Run Monte Carlo Simulation"):
                try:
                    r = result['portfolio_returns']
                    if r is None or len(r) < 5:
                        st.error("Insufficient return history for simulation")
                    else:
                        # Generate simulation (simplified version)
                        log_r = np.log1p(r.dropna().values)
                        mu_l = float(np.mean(log_r))
                        sigma_l = float(np.std(log_r, ddof=0))
                        T = int(round(years * TRADING_DAYS))
                        
                        rng = np.random.default_rng()
                        start_val = result['portfolio_value'].iloc[-1]
                        
                        if model.startswith("Bootstrap"):
                            increments = rng.choice(log_r, size=(T, n_paths), replace=True)
                        else:
                            increments = rng.normal(loc=mu_l, scale=sigma_l, size=(T, n_paths))
                        
                        log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)])
                        paths = start_val * np.exp(log_paths)
                        
                        # Store results
                        st.session_state.mc_result = {
                            'paths': paths,
                            'start_value': start_val,
                            'years': years,
                            'n_paths': n_paths,
                            'model': model
                        }
                        
                except Exception as e:
                    st.error(f"Error in Monte Carlo simulation: {str(e)}")
            
            if 'mc_result' in st.session_state:
                mc = st.session_state.mc_result
                paths = mc['paths']
                
                # Calculate statistics
                end_vals = paths[-1, :]
                end_rets = end_vals / mc['start_value'] - 1.0
                exp_return = float(np.mean(end_rets))
                med_return = float(np.median(end_rets))
                var95 = float(np.percentile(end_rets, 5))
                prob_loss = float(np.mean(end_rets < 0.0))
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expected Return", f"{exp_return*100:.2f}%")
                with col2:
                    st.metric("Median Return", f"{med_return*100:.2f}%")
                with col3:
                    st.metric("95% VaR", f"{var95*100:.2f}%")
                with col4:
                    st.metric("Probability of Loss", f"{prob_loss*100:.1f}%")
                
                # Plot results
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Paths
                show_paths = min(50, n_paths)
                for i in range(show_paths):
                    ax1.plot(paths[:, i], alpha=0.1, color='blue')
                
                # Percentiles
                q05 = np.percentile(paths, 5, axis=1)
                q50 = np.percentile(paths, 50, axis=1)
                q95 = np.percentile(paths, 95, axis=1)
                
                ax1.plot(q50, color='red', linewidth=2, label='Median')
                ax1.fill_between(range(len(q05)), q05, q95, alpha=0.3, color='red', label='5-95% Range')
                ax1.set_title(f"Monte Carlo Simulation ({mc['model']})")
                ax1.set_xlabel("Days")
                ax1.set_ylabel("Portfolio Value ($)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Distribution
                ax2.hist(end_rets * 100, bins=50, alpha=0.7, edgecolor='black')
                ax2.axvline(0, color='red', linestyle='--', label='Break-even')
                ax2.set_title("Ending Return Distribution")
                ax2.set_xlabel("Return (%)")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **Monte Carlo Interpretation:**
                - Simulates many possible future scenarios based on historical returns
                - **Median path**: Most likely outcome
                - **5-95% range**: Expected variability range
                - **Ending distribution**: Shows probability of different outcomes
                - Useful for understanding potential risks and rewards
                """)
        
        # Export functionality
        st.header("📤 Export Results")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to Excel"):
                mc_result = st.session_state.get('mc_result')
                excel_data = create_excel_export(result, mc_result)
                
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name=f"portfolio_backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            st.info("""
            **Excel Export Includes:**
            - Portfolio history with daily values
            - Complete performance metrics
            - Monte Carlo results (if available)
            - Ready for further analysis in Excel
            """)
    
    else:
        # Welcome screen when no backtest has been run
        st.info("""
        👈 **Configure your backtest in the sidebar and click 'Run Backtest' to get started.**
        
        **Available Features:**
        - Multi-asset portfolio backtesting
        - Various rebalancing strategies
        - Comprehensive risk metrics (Sharpe, Sortino, VaR, etc.)
        - Benchmark comparison
        - Transaction cost modeling
        - Rolling risk analysis
        - Correlation matrices
        - Monthly returns heatmaps
        - Monte Carlo simulations
        - Excel export functionality
        
        **Data Source**: Yahoo Finance (free historical data)
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'run_backtest' not in st.session_state:
        st.session_state.run_backtest = False
    
    main()
