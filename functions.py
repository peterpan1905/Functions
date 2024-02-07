def univariate(df):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  df_output = pd.DataFrame(columns=['type', 'missing', 'unique', 'min', 'q1', 'median',
                                    'q3', 'max', 'mode', 'mean', 'std', 'skew', 'kurt'])

  for col in df:
    # Features that apply to all dtypes
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    mode = df[col].mode()[0]
    if pd.api.types.is_numeric_dtype(df[col]):
      # Features for numeric only
      min = df[col].min()
      q1 = df[col].quantile(0.25)
      median = df[col].median()
      q3 = df[col].quantile(0.75)
      max = df[col].max()
      mean = df[col].mean()
      std = df[col].std()
      skew = df[col].skew()
      kurt = df[col].kurt()
      df_output.loc[col] = ["numeric", missing, unique, min, q1, median, q3, max, mode,
                            round(mean, 2), round(std, 2), round(skew, 2), round(kurt, 2)]
      sns.histplot(data=df, x=col)
      plt.show()
    else:
      df_output.loc[col] = ["categorical", missing, unique, '-', '-', '-', '-', '-',
                            mode, '-', '-', '-', '-']
      sns.countplot(data=df, x=col)
      plt.show()
  return df_output


def basic_wrangling(df, missing_threshold=0.50, unique_threshold=0.95, messages=True):
  import pandas as pd

  for col in df:
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    rows = df.shape[0]

    # Remove columns with too many unique values
    if missing / rows >= missing_threshold:
      df.drop(columns=[col], inplace=True)
      if messages: print(f"Column {col} dropped because of too much missing data ({round(missing / rows, 2) * 100}%)")
    # Remove columns with too much missing data
    elif unique / rows >= unique_threshold:
      if df[col].dtype in ['object', 'int64']:
        df.drop(columns=[col], inplace=True)
        if messages: print(f"Column {col} dropped because of too many unique values ({round(unique / rows, 2) * 100}%)")
    # Remove columns with single values
    elif unique == 1:
      df.drop(columns=[col], inplace=True)
      if messages: print(f"Column {col} dropped because there was only one value ({df[col].unique()[0]})")

  return df


def vif(df):
  import pandas as pd
  from sklearn.linear_model import LinearRegression

  df_vif = pd.DataFrame(columns=['vif'])

  for col in df:
    y = df[col]
    X = df.drop(columns=[col])
    r_squared = LinearRegression().fit(X, y).score(X, y)
    vif = 1 / (1 - r_squared)
    df_vif.loc[col] = vif

  return df_vif.sort_values(by=['vif'], ascending=False)