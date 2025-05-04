import pandas as pd

def preprocess_bitcoin(input_path: str):
    df = pd.read_csv(input_path)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['MTS'], unit='ms')
    
    # Create absolute trade sizes
    df['abs_amount'] = df['AMOUNT'].abs()
    
    return df[['timestamp', 'PRICE', 'AMOUNT', 'abs_amount']]

# Usage
df = preprocess_bitcoin('data/public_trade_data_btc_usd.csv')
df.to_csv('data/processed_trades.csv', index=False)