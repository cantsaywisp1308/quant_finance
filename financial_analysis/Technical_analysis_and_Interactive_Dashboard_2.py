import ipywidgets as wd
import cufflinks as cf
import pandas as pd
import yfinance as yf
from IPython.core.display_functions import display
from plotly.offline import iplot, init_notebook_mode

# Fix: newer numpy versions format np.float64 as 'np.float64(x)' in strings,
# which breaks cufflinks' rgba color generation. Patch to_rgba to use plain floats.
import cufflinks.colors as _cf_colors

def _patched_to_rgba(color, alpha):
    if type(color) == tuple:
        color, alpha = color
    color = color.lower()
    if 'rgba' in color:
        cl = list(eval(color.replace('rgba', '')))
        if alpha:
            cl[3] = float(alpha)
        r, g, b, a = int(cl[0]), int(cl[1]), int(cl[2]), float(cl[3])
        return f'rgba({r}, {g}, {b}, {a})'
    elif 'rgb' in color:
        r, g, b = eval(color.replace('rgb', ''))
        return f'rgba({int(r)}, {int(g)}, {int(b)}, {float(alpha)})'
    else:
        return _patched_to_rgba(_cf_colors.hex_to_rgb(color), alpha)

_cf_colors.to_rgba = _patched_to_rgba
# Patch in all modules that imported to_rgba directly
import cufflinks.plotlytools as _cf_pt
import cufflinks.tools as _cf_tools
_cf_pt.to_rgba = _patched_to_rgba
_cf_tools.to_rgba = _patched_to_rgba

cf.go_offline()
# Force offline mode: cf.go_offline() only sets the flag inside Jupyter (run_from_ipython()).
# When running as a plain Python script, we must set the flag directly.
import plotly.offline as _py_offline
_py_offline.__PLOTLY_OFFLINE_INITIALIZED = True
init_notebook_mode()

stocks = ["MSFT", "GOOGL", "FB", "TSLA", "AAPL"]
indicators = ["Bollinger Bands", "MACD", "RSI"]

def ta_dashboard(asset, indicators, start_date, end_date,
                 bb_k, bb_n, macd_fast, macd_slow, macd_signal,
                 rsi_periods, rsi_upper, rsi_lower):
    df = yf.download(asset,
                     start=start_date,
                     end=end_date,
                     progress=False,
                     auto_adjust=True)
    # Flatten MultiIndex columns returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    qf = cf.QuantFig(df, title= f'TA Dashboard - {asset}',
                     legend='right', name= f'{asset}')
    if 'Bollinger Bands' in indicators:
        qf.add_bollinger_bands(periods=bb_n, boll_std=bb_k)

    if 'MACD' in indicators:
        qf.add_macd(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)

    if 'RSI' in indicators:
        qf.add_rsi(periods=rsi_periods, rsi_upper=rsi_upper, rsi_lower=rsi_lower, showbands=True)

    return qf.iplot()

#define the selector group
def define_selectors(stocks, indicators, start_date, end_date):
    stocks_selector = wd.Dropdown(options=stocks, value= stocks[0], description='Stock')
    indicator_selector = wd.SelectMultiple(options=indicators, value= indicators, description='Indicator')
    start_date_selector = wd.DatePicker(description='Start Date', value= pd.to_datetime(start_date), continuous_update=False)
    end_date_selector = wd.DatePicker(description='End Date', value= pd.to_datetime(end_date), continuous_update=False)
    return stocks_selector, indicator_selector, start_date_selector, end_date_selector

#define the secondary selectors of the Bollinger Band
def define_bollinger_bands():
    bb_label = wd.Label('Bollinger Bands')
    n_param = wd.IntSlider(value=20, min=1, max=40, step=1, description='N:', continuous_update=False)
    k_param = wd.FloatSlider(value=2, min=0.5, max=4, step=0.5, description='k:', continuous_update=False)
    return bb_label, n_param, k_param

#define the MACD selectors
def define_macd_selectors():
    macd_label = wd.Label("MACD")

    macd_fast = wd.IntSlider(value=12, min=2, max=50, step=1,
                             description="Fast avg:",
                             continuous_update=False)

    macd_slow = wd.IntSlider(value=26, min=2, max=50, step=1,
                             description="Slow avg:",
                             continuous_update=False)

    macd_signal = wd.IntSlider(value=9, min=2, max=50, step=1,
                               description="MACD signal:",
                               continuous_update=False)
    return macd_label, macd_fast, macd_slow, macd_signal

def define_rsi_selectors():
    rsi_label = wd.Label("RSI")
    rsi_period = wd.IntSlider(value=14, min=2, max=50, step=1,
                              description="RSI period:",
                              continuous_update=False)
    rsi_upper = wd.FloatSlider(value=70, min=1, max=100, step=1,
                               description="RSI upper threshold:",
                               continuous_update=False)
    rsi_lower = wd.FloatSlider(value=30, min=1, max=100, step=1,
                               description="RSI lower threshold:",
                               continuous_update=False)
    return rsi_label, rsi_period, rsi_upper, rsi_lower

#define secondary group selector
def define_secondary_selectors():
    sec_selector_label = wd.Label("Secondary parameters",
                                  layout=wd.Layout(height="45px"))
    blank_label = wd.Label("", layout=wd.Layout(height="45px"))

    sec_box_1 = wd.VBox([sec_selector_label, bollinger_box, macd_box])
    sec_box_2 = wd.VBox([blank_label, rsi_box])
    return sec_box_1, sec_box_2

if __name__ == '__main__':
    main_selector_label = wd.Label('Main Selector', layout= wd.Layout(height='45px'))
    stock_selector, indicator_selector, start_date_selector, end_date_selector = define_selectors(stocks, indicators, '2018-01-01', '2018-12-31')
    main_selector_box = wd.VBox(children=[main_selector_label, stock_selector, indicator_selector, start_date_selector, end_date_selector])

    bb_label, n_param, k_param = define_bollinger_bands()
    bollinger_box = wd.VBox(children=[bb_label, n_param, k_param])

    macd_label, macd_fast ,macd_slow, macd_signal = define_macd_selectors()
    macd_box = wd.VBox(children=[macd_label, macd_fast, macd_slow, macd_signal])

    rsi_label, rsi_period, rsi_upper, rsi_lower = define_rsi_selectors()
    rsi_box = wd.VBox(children=[rsi_label, rsi_period,rsi_upper, rsi_lower])

    sec_box_1, sec_box_2 = define_secondary_selectors()
    secondary_selector_box = wd.HBox([sec_box_1, sec_box_2])

    controls_dict = {"asset": stock_selector,
                     "indicators": indicator_selector,
                     "start_date": start_date_selector,
                     "end_date": end_date_selector,
                     "bb_k": k_param,
                     "bb_n": n_param,
                     "macd_fast": macd_fast,
                     "macd_slow": macd_slow,
                     "macd_signal": macd_signal,
                     "rsi_periods": rsi_period,
                     "rsi_upper": rsi_upper,
                     "rsi_lower": rsi_lower}

    for asset in stocks:
        ui = wd.HBox([main_selector_box, secondary_selector_box])
        out = wd.interactive_output(ta_dashboard, controls_dict)
        display(ui, out)