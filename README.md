# Options Trading GUI Application

This is a Python-based application designed for options trading analysis using real-time data from Tastytrade and Schwab APIs. The app provides users with a graphical interface to visualize implied volatility smiles, analyze option chains, and filter options based on various parameters like bid-ask spread, strike range, and open interest.

## Requirements

Make sure you have the following Python libraries installed:

- numpy
- pandas
- pandas_datareader
- pykalman
- yfinance
- matplotlib
- mplfinance
- tqdm
- opencv-python

You can install these libraries by running the following command:

`pip install numpy pandas pandas_datareader pykalman yfinance matplotlib mplfinance tqdm opencv-python`

## Usage

1. Clone this repository and navigate to the project folder:

`git clone https://github.com/hedge0/OptionsTradingGui.git cd OptionsTradingGui`

2. Run the app using the following command:

`python app.py`

3. The application will open a GUI where you can select a platform (TastyTrade or Schwab), enter your credentials, and begin analyzing options data.

## Features

- **Platform Selection**: Choose between TastyTrade and Schwab for live options data streaming.
- **Implied Volatility Analysis**: Visualize IV smiles for selected options chains.
- **Custom Filters**: Filter options based on bid-ask spread, strike price, and open interest.
- **Fit Models**: Apply various models (RBF, RFV, SLV, SABR) to generate interpolations of the IV curve.

## License

This project is licensed under the MIT License.
