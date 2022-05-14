from kraken import coin, fiat, kraken
from datetime import date, datetime, timedelta
import pandas as pd
import requests
import warnings
import math

from python_json_config import ConfigBuilder

config_builder = ConfigBuilder()
config = config_builder.parse_config('config.json')
dict_config = config.to_dict()
DIR_TMP = config.sunrise_lib.DIR_TMP


class POW_Coin:
    def __init__(self, coin : coin):
        # TODO: figure out a way to use the abstract base class as a dispatcher for the specialized
        # subclasses/overridden methods based on the "coin" enum
        # TODO: implement daemon RPC calls using monero-python (pip install monero)
        now = datetime.now()
        self.coin = coin
        self.node_url = dict_config["profitability"][coin.name]["node"]
        self.blocktime = dict_config["profitability"][coin.name]["blocktime"]
        self._height = None
        self._height_last_fetched = now
        self._price = None
        self._price_last_fetched = now
        self._difficulty = None
        self._difficulty_last_fetched = now
        self._reward = None
        self._reward_last_fetched = now
    
    @property
    def price(self) -> float:
        now = datetime.now()
        if self._price is None or now - self._price_last_fetched > timedelta(minutes=config.profitability.refresh_delta_mins):
            self._price = kraken.get_prices(self.coin)
            self._price_last_fetched = now
        return self._price
    
    @property
    def difficulty(self) -> int:
        now = datetime.now()
        if self._difficulty is None or now - self._difficulty_last_fetched > timedelta(minutes=config.profitability.refresh_delta_mins):
            self.get_info()
        return self._difficulty
    
    @property
    def height(self) -> int:
        now = datetime.now()
        if self._height is None or now - self._height_last_fetched > timedelta(minutes=config.profitability.refresh_delta_mins):
            self.get_info()
        return self._height
    
    @property
    def reward(self) -> float:
        now = datetime.now()
        if self._reward is None or now - self._reward_last_fetched > timedelta(minutes=config.profitability.refresh_delta_mins):
            req_data = {"jsonrpc": "2.0", "id": "0", "method": "get_last_block_header"}
            data = requests.post(f"{self.node_url}/json_rpc", json=req_data).json()
            self._reward = int(data["result"]["block_header"]["reward"]) / 1e12
            self._reward_last_fetched = now
        return self._reward
    
    def profitability(self, fiat: fiat, hashrate: int, power_consumption: int, electricity_cost: float, pool_fee:float=0, price:float=None, reward:float=None, difficulty:int=None):
        if pool_fee < 0 or pool_fee > 1:
            raise ValueError("Invalid pool fee!")
        # nethash = self.difficulty / self.blocktime
        data = {
            "coin": self.coin.name,
            "difficulty": difficulty if difficulty else self.difficulty,
            "price": price if price else self.price[fiat],
            "reward": reward if reward else self.reward,
            "efficiency": hashrate / power_consumption if power_consumption != 0 else math.inf,
            "mining_cost_s": power_consumption * electricity_cost / 1000 / 3600,  # W * fiat/(kW*h) / 1000 W/kW / 3600 s/h = fiat/s
        }
        data["expected_crypto_income_s"] = data["reward"] * hashrate / data["difficulty"] * (1 - pool_fee)
        data["emission_rate_s"] = data["price"] * data["reward"] / self.blocktime
        data["expected_fiat_income_s"] = data["price"] * data["expected_crypto_income_s"]
        data["expected_fiat_income_kwh"] = data["efficiency"] * data["reward"] * data["price"] * (1 - pool_fee) * 1000 * 3600 / data["difficulty"]
        data["profitability"] = 100 * (data["expected_fiat_income_s"] - data["mining_cost_s"]) / data["mining_cost_s"] if data["mining_cost_s"] != 0 else math.inf
        data["breakeven_efficiency"] = (data["difficulty"] * electricity_cost) / (data["price"] * data["reward"] * 1000 * 3600 * (1 - pool_fee))
        return data
    
    def get_info(self) -> dict:
        now = datetime.now()
        json_req = { "jsonrpc": "2.0", "id": "0", "method": "get_info"}
        try:
            data = requests.get(f"{self.node_url}/json_rpc", json=json_req).json()
        except Exception as e:
            print("Error!", e)
            pass
        else:
            self._height = data["result"]["height"]
            self._height_last_fetched = now
            self._difficulty = data["result"]["difficulty"]
            self._difficulty_last_fetched = now
        return data["result"]
    
    def _request_headers_range(self, start_height:int, end_height:int) -> pd.DataFrame:
        json_req = {
            "jsonrpc": "2.0",
            "id": "0",
            "method": "get_block_headers_range",
            "params": {"start_height": start_height, "end_height": end_height}
        }
        data = requests.post(f"{self.node_url}/json_rpc", json=json_req).json()
        result = pd.DataFrame(data["result"]["headers"], columns=["height", "timestamp", "difficulty", "reward"])
        result.set_index("height", inplace=True)
        return result
    
    def _request_headers_batcher(self, start_height:int, end_height:int, batch_size:int=1000) -> pd.DataFrame:
        # if (end_height - start_height) >= batch_size, send a batch request then
        # evaluate again with start_height = start_height + batch_size
        # if it's false request the last range from start_height to end_height
        # TODO: handle KeyboardCancellation to save partial data
        results = []
        while (end_height - start_height >= batch_size):
            print(f"Downloading headers from {start_height} to {start_height + batch_size - 1}")
            results.append(self._request_headers_range(start_height, start_height + batch_size - 1))
            start_height = start_height + batch_size
        print(f"Downloading headers from {start_height} to {end_height}")
        results.append(self._request_headers_range(start_height, end_height))
        result = pd.concat(results, axis=0)
        return result
    
    def historical_diff(self, height:int=None, timestamp:datetime=None, batch_size:int=1000) -> int:
        if height and timestamp:
            warnings.warn("Both height and timestamp present: ignoring timestamp", UserWarning)
        
        path = f"{DIR_TMP}/diff_{self.coin.name}.pkl"
        try: # If we have previous saved data, merge with the new data
            diff = pd.read_pickle(path)
        except FileNotFoundError:
            print(f"{path} does not exist. Creating...")
            # diff = self._request_headers_batcher(0, self.height, batch_size=batch_size)
            diff = self._request_headers_batcher(0, height, batch_size=batch_size)
            diff.to_pickle(path)
            print(diff)
            pass
        
        last_known_height = int(diff.index[-1])
        last_known_timestamp = datetime.fromtimestamp(diff["timestamp"].iloc[-1])
        # print("last_known_timestamp", last_known_timestamp)
        
        if height:
            if height > self.height:
                raise ValueError("Requested height is in the future")
            if height < 0:
                raise ValueError("Cannot have a negative height")
            if height > last_known_height:  # Need to update
                # diff_new = self._request_headers_batcher(last_known_block + 1, self.height, batch_size=batch_size)
                diff_new = self._request_headers_batcher(last_known_height + 1, height, batch_size=batch_size)
                diff = pd.concat([diff, diff_new], axis=0)
                diff.to_pickle(path)
            return diff.at[height, "difficulty"]
        elif timestamp:
            dt_timestamp = datetime.fromtimestamp(timestamp)
            # print("dt_timestamp", dt_timestamp)
            if dt_timestamp > datetime.now():
                raise ValueError("Requested timestamp is in the future")
            if dt_timestamp < datetime.fromtimestamp(0):
                raise ValueError("Cannot have a negative timestamp")
            if dt_timestamp > last_known_timestamp:  # Need to update
                xmr_blocktime_120_height = 1009827
                xmr_blocktime_120_ts = 1458748658
                xmr_blocktime_120_dt = datetime.fromtimestamp(xmr_blocktime_120_ts)
                if dt_timestamp < xmr_blocktime_120_dt:
                    # print((dt_timestamp - last_known_timestamp).total_seconds() / 60)
                    target_height = min(self.height, last_known_height + math.ceil((dt_timestamp - last_known_timestamp).total_seconds() / 60))
                else:
                    # print((dt_timestamp - xmr_blocktime_120_dt).total_seconds() / 120)
                    target_height = min(self.height, xmr_blocktime_120_height + math.ceil((dt_timestamp - xmr_blocktime_120_dt).total_seconds() / 120))
                # print("target_height", target_height)
                # diff_new = self._request_headers_batcher(last_known_height + 1, self.height, batch_size=batch_size)
                diff_new = self._request_headers_batcher(last_known_height + 1, target_height, batch_size=batch_size)
                diff = pd.concat([diff, diff_new], axis=0)
                diff.to_pickle(path)
            # search timestamps and find nearest-previous height index
            # res = diff.loc[diff.index.get_loc(timestamp, method="nearest")]  # get_loc is deprecated
            res = diff.loc[diff["timestamp"] <= timestamp]
            # print(res.at[res.last_valid_index(), "timestamp"])
            # print(res["difficulty"].iloc[-1])
            return res["difficulty"].iloc[-1]
        else:  # not height and not timestamp:
            raise TypeError("Need a height or a timestamp")


def test():
    import time
    a = POW_Coin(coin.XMR)
    b = a.profitability(fiat.USD, 20000, 200, 0.1)
    print(b)
    print(a.difficulty)
    print(a._difficulty_last_fetched)
    time.sleep(2)
    print(a.height)
    print(a._height_last_fetched)  # Must match a._difficulty_last_fetched
    print(a.get_info())
    # h1 = a._request_headers_range(2500000,2500004)
    # h2 = a._request_headers_range(2500003,2500005)
    # h3 = pd.concat([h1, h2], axis=0)
    # h4 = h1.combine_first(h2)
    # print(h1)
    # print(h2)
    # print(h3)
    # print(h4)
    # h5 = a._request_headers_batcher(2500000, 2500001, batch_size=1)
    # print(h5)
    # print(a.historical_diff(height=10))
    print(a.historical_diff(timestamp=1397818225))
    print(a.historical_diff(timestamp=1447969434))
    print(a.historical_diff(timestamp=1497818200))

if __name__ == "__main__":
    test()
