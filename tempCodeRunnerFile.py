    df['sunrise'] = pd.to_datetime(df['sunrise'])
    df['sunset'] = pd.to_datetime(df['sunset'])
    df['moonrise'] = pd.to_datetime(df['moonrise'])
    df['moonset'] = pd.to_datetime(df['moonset'])