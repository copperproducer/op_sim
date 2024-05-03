import pandas as pd
import numpy as np
from scipy.stats import norm


# Load the CSV file
file_path = 'simulation_data.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter the data between start_date and end_date
start_date = pd.Timestamp('2024-01-15')
end_date = pd.Timestamp('2024-04-28')
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Check the first few rows to understand the available columns
filtered_data.head(), filtered_data.columns

# Define the Black-Scholes formula for a call option
def black_scholes_call(S, K, T, r, sigma):
    """
    S: stock price
    K: strike price
    T: time to maturity in years
    r: risk-free rate
    sigma: volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    S: stock price
    K: strike price
    T: time to maturity in years
    r: risk-free rate
    sigma: volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

# Constants
risk_free_rate = 0.06  # 6% annual risk-free rate



def option1():
    # Calculate the initial TRT price and volatility on the start date
    initial_trt_data = filtered_data.dropna(subset=['TRT_price', 'TRT_predicted_volatility']).iloc[0]
    initial_trt_price = initial_trt_data['TRT_price']
    initial_trt_volatility = initial_trt_data['TRT_predicted_volatility']
    strike_price = initial_trt_price * 1.20  # 20% OTM
    option_expiration_date = min(end_date, start_date + pd.DateOffset(months=3))

    # Calculate the days to expiration
    days_to_expiration = (option_expiration_date - start_date).days / 365

    # Calculate the option price using Black-Scholes model
    option_price = black_scholes_call(
        S=initial_trt_price,
        K=strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_trt_volatility
    )

    # Prepare the results
    summary = {
        "Initial TRT Price": initial_trt_price,
        "Strike Price": strike_price,
        "Option Price": option_price,
        "Breakeven Price": strike_price + option_price,
        "Expiration Date": option_expiration_date
    }






    # Find the TRT price on or immediately before the expiration date
    expiration_data_corrected = filtered_data[(filtered_data['Date'] <= option_expiration_date) & filtered_data['TRT_price'].notna()].iloc[-1]

    final_trt_price_corrected = expiration_data_corrected['TRT_price']
    expiration_date_actual_corrected = expiration_data_corrected['Date']

    # Recalculate profit or loss with correct dates
    if final_trt_price_corrected > strike_price:
        profit_corrected = max(final_trt_price_corrected - strike_price, 0) * 100 - option_price * 100
    else:
        profit_corrected = -option_price * 100

    # Update summary with corrected results
    summary.update({
        "Final TRT Price on Expiration": final_trt_price_corrected,
        "Actual Expiration Date": expiration_date_actual_corrected,
        "Profit/Loss ($)": profit_corrected
    })

    print(summary)

def option2():
    # Calculate the initial QQQ price and volatility on the start date
    initial_qqq_data = filtered_data.dropna(subset=['QQQ_price', 'QQQ_predicted_volatility']).iloc[0]
    initial_qqq_price = initial_qqq_data['QQQ_price']
    initial_qqq_volatility = initial_qqq_data['QQQ_predicted_volatility']
    qqq_strike_price = initial_qqq_price * 0.80  # 20% ITM
    qqq_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))

    # Calculate the days to expiration for QQQ
    qqq_days_to_expiration = (qqq_option_expiration_date - start_date).days / 365

    # Calculate the QQQ option price using Black-Scholes model
    qqq_option_price = black_scholes_call(
        S=initial_qqq_price,
        K=qqq_strike_price,
        T=qqq_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )

    # Prepare the QQQ results
    qqq_summary = {
        "Initial QQQ Price": initial_qqq_price,
        "Strike Price": qqq_strike_price,
        "Option Price": qqq_option_price,
        "Breakeven Price": qqq_strike_price + qqq_option_price,
        "Expiration Date": qqq_option_expiration_date
    }



        # Find the QQQ price on or immediately before the expiration date for the option
    qqq_expiration_data = filtered_data[(filtered_data['Date'] <= qqq_option_expiration_date) & filtered_data['QQQ_price'].notna()].iloc[-1]

    final_qqq_price = qqq_expiration_data['QQQ_price']
    qqq_expiration_date_actual = qqq_expiration_data['Date']

    # Recalculate profit or loss with correct dates for QQQ
    if final_qqq_price > qqq_strike_price:
        qqq_profit = max(final_qqq_price - qqq_strike_price, 0) * 100 - qqq_option_price * 100
    else:
        qqq_profit = -qqq_option_price * 100

    # Update QQQ summary with results
    qqq_summary.update({
        "Final QQQ Price on Expiration": final_qqq_price,
        "Actual Expiration Date": qqq_expiration_date_actual,
        "Profit/Loss ($)": qqq_profit
    })

    print(qqq_summary)

def sell_covered_call_on_wba():
    # Calculate the initial WBA price and volatility on the start date
    initial_wba_data = filtered_data.dropna(subset=['WBA_price', 'WBA_predicted_volatility']).iloc[0]
    initial_wba_price = initial_wba_data['WBA_price']
    initial_wba_volatility = initial_wba_data['WBA_predicted_volatility']
    wba_strike_price = initial_wba_price * 0.80  # 20% ITM
    wba_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))

    # Calculate the days to expiration for WBA
    wba_days_to_expiration = (wba_option_expiration_date - start_date).days / 365

    # Calculate the WBA call option price using Black-Scholes model
    wba_option_price = black_scholes_call(
        S=initial_wba_price,
        K=wba_strike_price,
        T=wba_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_wba_volatility
    )

    # Calculate the maximum profit
    max_profit = (wba_option_price + (wba_strike_price - initial_wba_price)) * 100

    # Prepare the WBA results
    wba_summary = {
        "Initial WBA Price": initial_wba_price,
        "Strike Price": wba_strike_price,
        "Option Price": wba_option_price,
        "Breakeven Price": wba_strike_price - wba_option_price,
        "Max Profit": max_profit,
        "Expiration Date": wba_option_expiration_date
    }

    # Find the WBA price on or immediately before the expiration date for the option
    wba_expiration_data = filtered_data[(filtered_data['Date'] <= wba_option_expiration_date) & filtered_data['WBA_price'].notna()].iloc[-1]

    final_wba_price = wba_expiration_data['WBA_price']
    wba_expiration_date_actual = wba_expiration_data['Date']

    # Calculate profit or loss
    if final_wba_price >= wba_strike_price:
        wba_profit_loss = max_profit
    else:
        wba_profit_loss = (final_wba_price - initial_wba_price) * 100

    # Update WBA summary with results
    wba_summary.update({
        "Final WBA Price on Expiration": final_wba_price,
        "Actual Expiration Date": wba_expiration_date_actual,
        "Profit/Loss ($)": wba_profit_loss
    })

    print(wba_summary)


    

def buy_put_option_on_bby():
    # Calculate the initial BBY price and volatility on the start date
    initial_bby_data = filtered_data.dropna(subset=['BBY_price', 'BBY_predicted_volatility']).iloc[0]
    initial_bby_price = initial_bby_data['BBY_price']
    initial_bby_volatility = initial_bby_data['BBY_predicted_volatility']
    bby_strike_price = initial_bby_price
    bby_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))

    # Calculate the days to expiration for BBY
    bby_days_to_expiration = (bby_option_expiration_date - start_date).days / 365

    # Calculate the BBY put option price using Black-Scholes model
    bby_option_price = black_scholes_put(
        S=initial_bby_price,
        K=bby_strike_price,
        T=bby_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_bby_volatility
    )

    # Prepare the BBY results
    bby_summary = {
        "Initial BBY Price": initial_bby_price,
        "Strike Price": bby_strike_price,
        "Option Price": bby_option_price,
        "Breakeven Price": bby_strike_price - bby_option_price,
        "Expiration Date": bby_option_expiration_date
    }

    # Find the BBY price on or immediately before the expiration date for the option
    bby_expiration_data = filtered_data[(filtered_data['Date'] <= bby_option_expiration_date) & filtered_data['BBY_price'].notna()].iloc[-1]

    final_bby_price = bby_expiration_data['BBY_price']
    bby_expiration_date_actual = bby_expiration_data['Date']

    # Calculate profit or loss
    if final_bby_price < bby_strike_price:
        bby_profit_loss = max(bby_strike_price - final_bby_price, 0) * 100 - bby_option_price * 100
    else:
        bby_profit_loss = -bby_option_price * 100

    # Update BBY summary with results
    bby_summary.update({
        "Final BBY Price on Expiration": final_bby_price,
        "Actual Expiration Date": bby_expiration_date_actual,
        "Profit/Loss ($)": bby_profit_loss
    })

    print(bby_summary)


def buy_naked_put_on_dltr():
    # Calculate the initial DLTR price and volatility on the start date
    initial_dltr_data = filtered_data.dropna(subset=['DLTR_price', 'DLTR_predicted_volatility']).iloc[0]
    initial_dltr_price = initial_dltr_data['DLTR_price']
    initial_dltr_volatility = initial_dltr_data['DLTR_predicted_volatility']
    dltr_strike_price = initial_dltr_price
    dltr_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))

    # Calculate the days to expiration for DLTR
    dltr_days_to_expiration = (dltr_option_expiration_date - start_date).days / 365

    # Calculate the DLTR put option price using Black-Scholes model
    dltr_option_price = black_scholes_put(
        S=initial_dltr_price,
        K=dltr_strike_price,
        T=dltr_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_dltr_volatility
    )

    # Prepare the DLTR results
    dltr_summary = {
        "Initial DLTR Price": initial_dltr_price,
        "Strike Price": dltr_strike_price,
        "Option Price": dltr_option_price,
        "Breakeven Price": dltr_strike_price - dltr_option_price,
        "Expiration Date": dltr_option_expiration_date
    }

    # Find the DLTR price on or immediately before the expiration date for the option
    dltr_expiration_data = filtered_data[(filtered_data['Date'] <= dltr_option_expiration_date) & filtered_data['DLTR_price'].notna()].iloc[-1]

    final_dltr_price = dltr_expiration_data['DLTR_price']
    dltr_expiration_date_actual = dltr_expiration_data['Date']

    # Calculate profit or loss
    if final_dltr_price < dltr_strike_price:
        dltr_profit_loss = max(dltr_strike_price - final_dltr_price, 0) * 100 - dltr_option_price * 100
    else:
        dltr_profit_loss = -dltr_option_price * 100

    # Update DLTR summary with results
    dltr_summary.update({
        "Final DLTR Price on Expiration": final_dltr_price,
        "Actual Expiration Date": dltr_expiration_date_actual,
        "Profit/Loss ($)": dltr_profit_loss
    })

    print(dltr_summary)

def buy_protective_put_on_wmt(max_loss):
    # Calculate the initial WMT price and volatility on the start date
    initial_wmt_data = filtered_data.dropna(subset=['WMT_price', 'WMT_predicted_volatility']).iloc[0]
    initial_wmt_price = initial_wmt_data['WMT_price']
    initial_wmt_volatility = initial_wmt_data['WMT_predicted_volatility']
    wmt_strike_price = initial_wmt_price
    wmt_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))

    # Calculate the days to expiration for WMT
    wmt_days_to_expiration = (wmt_option_expiration_date - start_date).days / 365

    # Calculate the WMT put option price using Black-Scholes model
    wmt_option_price = black_scholes_put(
        S=initial_wmt_price,
        K=wmt_strike_price,
        T=wmt_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_wmt_volatility
    )

    # Calculate the number of put contracts needed
    num_contracts = int(np.ceil(max_loss / (wmt_option_price * 100)))

    # Calculate the total cost of buying the put contracts
    total_cost = wmt_option_price * num_contracts * 100

    # Prepare the WMT results
    wmt_summary = {
        "Initial WMT Price": initial_wmt_price,
        "Strike Price": wmt_strike_price,
        "Option Price": wmt_option_price,
        "Breakeven Price": wmt_strike_price - wmt_option_price,
        "Expiration Date": wmt_option_expiration_date,
        "Max Loss": max_loss,
        "Number of Put Contracts Needed": num_contracts,
        "Total Cost": total_cost
    }

    # Find the WMT price on or immediately before the expiration date for the option
    wmt_expiration_data = filtered_data[(filtered_data['Date'] <= wmt_option_expiration_date) & filtered_data['WMT_price'].notna()].iloc[-1]

    final_wmt_price = wmt_expiration_data['WMT_price']
    wmt_expiration_date_actual = wmt_expiration_data['Date']

    # Calculate profit or loss
    if final_wmt_price < wmt_strike_price:
        wmt_profit_loss = max(wmt_strike_price - final_wmt_price, 0) * 100 - total_cost
    else:
        wmt_profit_loss = -total_cost

    # Update WMT summary with results
    wmt_summary.update({
        "Final WMT Price on Expiration": final_wmt_price,
        "Actual Expiration Date": wmt_expiration_date_actual,
        "Profit/Loss ($)": wmt_profit_loss
    })

    print(wmt_summary)

def sell_straddle_on_qqq():
    # Calculate the initial QQQ price and volatility on the start date
    initial_qqq_data = filtered_data.dropna(subset=['QQQ_price', 'QQQ_predicted_volatility']).iloc[0]
    initial_qqq_price = initial_qqq_data['QQQ_price']
    initial_qqq_volatility = initial_qqq_data['QQQ_predicted_volatility']
    
    # Calculate the strike prices for the call and put options
    call_strike_price = initial_qqq_price * 1.20  # 20% above initial QQQ price
    put_strike_price = initial_qqq_price * 0.80  # 20% below initial QQQ price
    
    # Calculate the option expiration date
    option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))
    
    # Calculate the days to expiration for the options
    days_to_expiration = (option_expiration_date - start_date).days / 365
    
    # Calculate the call and put option prices using Black-Scholes model
    call_option_price = black_scholes_call(
        S=initial_qqq_price,
        K=call_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )
    
    put_option_price = black_scholes_put(
        S=initial_qqq_price,
        K=put_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )
    
    # Calculate the total premium received from selling the straddle
    total_premium_received = (call_option_price + put_option_price) * 100  # Assuming one contract each
    
    # Prepare the QQQ straddle results
    qqq_straddle_summary = {
        "Initial QQQ Price": initial_qqq_price,
        "Call Strike Price": call_strike_price,
        "Put Strike Price": put_strike_price,
        "Call Option Price": call_option_price,
        "Put Option Price": put_option_price,
        "Option Expiration Date": option_expiration_date,
        "Total Premium Received": total_premium_received
    }
    
    print(qqq_straddle_summary)


def long_money_spread_on_dltr():
    # Calculate the initial DLTR price and volatility on the start date
    initial_dltr_data = filtered_data.dropna(subset=['DLTR_price', 'DLTR_predicted_volatility']).iloc[0]
    initial_dltr_price = initial_dltr_data['DLTR_price']
    initial_dltr_volatility = initial_dltr_data['DLTR_predicted_volatility']
    
    # Calculate the strike prices for the long and short options
    long_strike_price = initial_dltr_price * 0.90  # 10% below initial DLTR price
    short_strike_price = initial_dltr_price * 0.95  # 5% below initial DLTR price
    
    # Calculate the option expiration date
    option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))
    
    # Calculate the days to expiration for the options
    days_to_expiration = (option_expiration_date - start_date).days / 365
    
    # Calculate the long and short option prices using Black-Scholes model
    long_option_price = black_scholes_call(
        S=initial_dltr_price,
        K=long_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_dltr_volatility
    )
    
    short_option_price = black_scholes_call(
        S=initial_dltr_price,
        K=short_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_dltr_volatility
    )
    
    # Calculate the net debit or credit of the spread
    net_debit_credit = long_option_price - short_option_price
    
    # Prepare the DLTR spread results
    dltr_spread_summary = {
        "Initial DLTR Price": initial_dltr_price,
        "Long Strike Price": long_strike_price,
        "Short Strike Price": short_strike_price,
        "Long Option Price": long_option_price,
        "Short Option Price": short_option_price,
        "Option Expiration Date": option_expiration_date,
        "Net Debit/Credit": net_debit_credit
    }
    
    print(dltr_spread_summary)

def bear_money_spread_on_bby():
    # Calculate the initial BBY price and volatility on the start date
    initial_bby_data = filtered_data.dropna(subset=['BBY_price', 'BBY_predicted_volatility']).iloc[0]
    initial_bby_price = initial_bby_data['BBY_price']
    initial_bby_volatility = initial_bby_data['BBY_predicted_volatility']
    
    # Calculate the strike prices for the long and short options
    long_strike_price = initial_bby_price * 1.05  # 5% above initial BBY price
    short_strike_price = initial_bby_price * 1.10  # 10% above initial BBY price
    
    # Calculate the option expiration date
    option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))
    
    # Calculate the days to expiration for the options
    days_to_expiration = (option_expiration_date - start_date).days / 365
    
    # Calculate the long and short option prices using Black-Scholes model
    long_option_price = black_scholes_put(
        S=initial_bby_price,
        K=long_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_bby_volatility
    )
    
    short_option_price = black_scholes_put(
        S=initial_bby_price,
        K=short_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_bby_volatility
    )
    
    # Calculate the net debit or credit of the spread
    net_debit_credit = short_option_price - long_option_price
    
    # Prepare the BBY spread results
    bby_spread_summary = {
        "Initial BBY Price": initial_bby_price,
        "Long Strike Price": long_strike_price,
        "Short Strike Price": short_strike_price,
        "Long Option Price": long_option_price,
        "Short Option Price": short_option_price,
        "Option Expiration Date": option_expiration_date,
        "Net Debit/Credit": net_debit_credit
    }
    
    print(bby_spread_summary)

def zero_cost_collar_on_dal():
    # Calculate the initial DAL price and volatility on the start date
    initial_dal_data = filtered_data.dropna(subset=['DAL_price', 'DAL_predicted_volatility']).iloc[0]
    initial_dal_price = initial_dal_data['DAL_price']
    initial_dal_volatility = initial_dal_data['DAL_predicted_volatility']
    
    # Calculate the strike price for the put option
    put_strike_price = initial_dal_price * 0.95  # 5% below initial DAL price
    
    # Calculate the strike price for the call option
    call_strike_price = initial_dal_price * 1.05  # 5% above initial DAL price
    
    # Calculate the option expiration date
    option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))
    
    # Calculate the days to expiration for the options
    days_to_expiration = (option_expiration_date - start_date).days / 365
    
    # Calculate the put option price using Black-Scholes model
    put_option_price = black_scholes_put(
        S=initial_dal_price,
        K=put_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_dal_volatility
    )
    
    # Calculate the call option price using Black-Scholes model
    call_option_price = black_scholes_call(
        S=initial_dal_price,
        K=call_strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_dal_volatility
    )
    
    # Calculate the number of put contracts needed to make the collar zero-cost
    num_put_contracts = int(np.ceil(call_option_price / (put_option_price * 100)))
    
    # Calculate the total cost of buying the protective put
    total_put_cost = put_option_price * num_put_contracts * 100
    
    # Calculate the total premium received from selling the covered call
    total_call_premium = call_option_price * num_put_contracts * 100
    
    # Calculate the breakeven price
    breakeven_price = initial_dal_price - total_put_cost + total_call_premium
    
    # Prepare the DAL collar spread results
    dal_collar_summary = {
        "Initial DAL Price": initial_dal_price,
        "Put Strike Price": put_strike_price,
        "Call Strike Price": call_strike_price,
        "Put Option Price": put_option_price,
        "Call Option Price": call_option_price,
        "Option Expiration Date": option_expiration_date,
        "Number of Put Contracts": num_put_contracts,
        "Total Cost of Buying Put": total_put_cost,
        "Total Premium Received from Selling Call": total_call_premium,
        "Breakeven Price": breakeven_price
    }
    
    print(dal_collar_summary)



def long_calendar_spread_on_qqq():
    # Calculate the initial QQQ price and volatility on the start date
    initial_qqq_data = filtered_data.dropna(subset=['QQQ_price', 'QQQ_predicted_volatility']).iloc[0]
    initial_qqq_price = initial_qqq_data['QQQ_price']
    initial_qqq_volatility = initial_qqq_data['QQQ_predicted_volatility']
    
    # Calculate the strike price for the options
    strike_price = initial_qqq_price * 1.05  # 5% above initial QQQ price
    
    # Calculate the expiration dates for the long and short options
    long_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=3))  # Longer-term option
    short_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))  # Shorter-term option
    
    # Calculate the days to expiration for the options
    long_days_to_expiration = (long_option_expiration_date - start_date).days / 365
    short_days_to_expiration = (short_option_expiration_date - start_date).days / 365
    
    # Calculate the long and short option prices using Black-Scholes model
    long_option_price = black_scholes_call(
        S=initial_qqq_price,
        K=strike_price,
        T=long_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )
    
    short_option_price = black_scholes_call(
        S=initial_qqq_price,
        K=strike_price,
        T=short_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )
    
    # Calculate the net debit or credit of the spread
    net_debit_credit = long_option_price - short_option_price
    
    # Prepare the QQQ calendar spread results
    qqq_calendar_summary = {
        "Initial QQQ Price": initial_qqq_price,
        "Strike Price": strike_price,
        "Long Option Expiration Date": long_option_expiration_date,
        "Short Option Expiration Date": short_option_expiration_date,
        "Long Option Price": long_option_price,
        "Short Option Price": short_option_price,
        "Net Debit/Credit": net_debit_credit
    }
    
def long_calendar_spread_on_qqq():
    # Calculate the initial QQQ price and volatility on the start date
    initial_qqq_data = filtered_data.dropna(subset=['QQQ_price', 'QQQ_predicted_volatility']).iloc[0]
    initial_qqq_price = initial_qqq_data['QQQ_price']
    initial_qqq_volatility = initial_qqq_data['QQQ_predicted_volatility']
    
    # Calculate the strike price for the options
    strike_price = initial_qqq_price * 1.05  # 5% above initial QQQ price
    
    # Calculate the expiration dates for the long and short options
    long_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=3))  # Longer-term option
    short_option_expiration_date = min(end_date, start_date + pd.DateOffset(months=1))  # Shorter-term option
    
    # Calculate the days to expiration for the options
    long_days_to_expiration = (long_option_expiration_date - start_date).days / 365
    short_days_to_expiration = (short_option_expiration_date - start_date).days / 365
    
    # Calculate the long and short option prices using Black-Scholes model
    long_option_price = black_scholes_call(
        S=initial_qqq_price,
        K=strike_price,
        T=long_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )
    
    short_option_price = black_scholes_call(
        S=initial_qqq_price,
        K=strike_price,
        T=short_days_to_expiration,
        r=risk_free_rate,
        sigma=initial_qqq_volatility
    )
    
    # Calculate the net debit or credit of the spread
    net_debit_credit = long_option_price - short_option_price
    
    # Prepare the QQQ calendar spread results
    qqq_calendar_summary = {
        "Initial QQQ Price": initial_qqq_price,
        "Strike Price": strike_price,
        "Long Option Expiration Date": long_option_expiration_date,
        "Short Option Expiration Date": short_option_expiration_date,
        "Long Option Price": long_option_price,
        "Short Option Price": short_option_price,
        "Net Debit/Credit": net_debit_credit
    }
    
    print(qqq_calendar_summary)

def delta_hedging_on_dltr():
    # Constants
    trading_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days between start and end dates
    small_delta_threshold = 0.05  # Threshold for delta change
    
    # Initialize variables
    portfolio_value = 0
    initial_dltr_data = filtered_data.dropna(subset=['DLTR_price', 'DLTR_predicted_volatility']).iloc[0]
    initial_dltr_price = initial_dltr_data['DLTR_price']
    initial_dltr_volatility = initial_dltr_data['DLTR_predicted_volatility']
    strike_price = initial_dltr_price  # Assuming ATM option
    option_expiration_date = min(end_date, start_date + pd.DateOffset(weeks=2))
    days_to_expiration = (option_expiration_date - start_date).days / 365
    option_price = black_scholes_call(
        S=initial_dltr_price,
        K=strike_price,
        T=days_to_expiration,
        r=risk_free_rate,
        sigma=initial_dltr_volatility
    )
    delta = norm.cdf((np.log(initial_dltr_price / strike_price) + (risk_free_rate + 0.5 * initial_dltr_volatility ** 2) * days_to_expiration) / (initial_dltr_volatility * np.sqrt(days_to_expiration)))

    # Log for trades
    trade_log = []

    for day in trading_days:
        # Get DLTR data for the day
        dltr_data = filtered_data[filtered_data['Date'] == day]
        if dltr_data.empty:
            continue  # Skip if no data for the day
        
        dltr_price = dltr_data['DLTR_price'].values[0]
        dltr_volatility = dltr_data['DLTR_predicted_volatility'].values[0]
        
        # Calculate new delta
        new_delta = norm.cdf((np.log(dltr_price / strike_price) + (risk_free_rate + 0.5 * dltr_volatility ** 2) * days_to_expiration) / (dltr_volatility * np.sqrt(days_to_expiration)))
        
        # Check delta change
        delta_change = abs(new_delta - delta)
        
        if delta_change > small_delta_threshold:
            # Rebalance the portfolio
            delta_difference = new_delta - delta
            shares_to_buy_sell = int(delta_difference * 100)  # Assuming one option contract covers 100 shares
            
            # Calculate the cost of buying/selling shares
            cost_of_shares = shares_to_buy_sell * dltr_price
            
            # Update portfolio value
            portfolio_value -= cost_of_shares
            
            # Log the trade
            trade_log.append(f"Date: {day}, Delta Change: {delta_change}, Delta: {new_delta}, Shares to Buy/Sell: {shares_to_buy_sell}, Cost: {cost_of_shares}")
            
            # Update delta
            delta = new_delta
        else:
            # Log when no trade is made due to small delta change
            trade_log.append(f"Date: {day}, Delta Change: {delta_change} (No Trade)")

    # Summary
    summary = {
        "Initial Delta": delta,
        "Final Delta": new_delta,
        "Portfolio Value Change": portfolio_value,
        "Number of Trades": len(trade_log),
        "Trade Log": trade_log
    }

    print(summary)



