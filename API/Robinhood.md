# Robinhood Trading API (1.0.0)

## Introduction
Welcome to Robinhood Crypto API documentation for traders and developers! The APIs let you view crypto market data, access your account information, and place crypto orders programmatically.

Interested in using our API? Get started with creating your keys on the Crypto Account Settings Page, available only on a desktop web browser.

Your use of the Robinhood Crypto API is subject to the Robinhood Crypto Customer Agreement as well as all other terms and disclosures made available on Robinhood Crypto's about page.

## Getting Started
Developing your own application to place trades with your Robinhood account is quick and simple. Start here with the code you'll need to access the API, authenticate credentials, and make supported API calls. These are essentially the building blocks of code for each API call, which you can easily build on based on your preferred strategies.

1. Create a script file
```
mkdir robinhood-api-trading && cd robinhood-api-trading
touch robinhood_api_trading.py
```

2. Install PyNaCl library
 ```
 pip install pynacl
 ```

3. Copy the script below into the newly created robinhood_api_trading.py file. Make sure to add your API key and secret key into the API_KEY and BASE64_PRIVATE_KEY variables.

```python
import base64
import datetime
import json
from typing import Any, Dict, Optional
import uuid
import requests
from nacl.signing import SigningKey

API_KEY = "ADD YOUR API KEY HERE"
BASE64_PRIVATE_KEY = "ADD YOUR PRIVATE KEY HERE"

class CryptoAPITrading:
    def __init__(self):
        self.api_key = API_KEY
        private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
        self.private_key = SigningKey(private_key_seed)
        self.base_url = "https://trading.robinhood.com"

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def get_query_params(key: str, *args: Optional[str]) -> str:
        if not args:
            return ""

        params = []
        for arg in args:
            params.append(f"{key}={arg}")

        return "?" + "&".join(params)

    def make_api_request(self, method: str, path: str, body: str = "") -> Any:
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        try:
            response = {}
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=json.loads(body), timeout=10)
            return response.json()
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # all supported symbols will be returned
    def get_trading_pairs(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/trading/trading_pairs/{query_params}"
        return self.make_api_request("GET", path)

    # The asset_codes argument must be formatted as the short form name for a crypto, e.g "BTC", "ETH". If no asset
    # codes are provided, all crypto holdings will be returned
    def get_holdings(self, *asset_codes: Optional[str]) -> Any:
        query_params = self.get_query_params("asset_code", *asset_codes)
        path = f"/api/v1/crypto/trading/holdings/{query_params}"
        return self.make_api_request("GET", path)

    # The symbols argument must be formatted in trading pairs, e.g "BTC-USD", "ETH-USD". If no symbols are provided,
    # the best bid and ask for all supported symbols will be returned
    def get_best_bid_ask(self, *symbols: Optional[str]) -> Any:
        query_params = self.get_query_params("symbol", *symbols)
        path = f"/api/v1/crypto/marketdata/best_bid_ask/{query_params}"
        return self.make_api_request("GET", path)

    # The symbol argument must be formatted in a trading pair, e.g "BTC-USD", "ETH-USD"
    # The side argument must be "bid", "ask", or "both".
    # Multiple quantities can be specified in the quantity argument, e.g. "0.1,1,1.999".
    def get_estimated_price(self, symbol: str, side: str, quantity: str) -> Any:
        path = f"/api/v1/crypto/marketdata/estimated_price/?symbol={symbol}&side={side}&quantity={quantity}"
        return self.make_api_request("GET", path)

    def place_order(
            self,
            client_order_id: str,
            side: str,
            order_type: str,
            symbol: str,
            order_config: Dict[str, str],
    ) -> Any:
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            f"{order_type}_order_config": order_config,
        }
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("POST", path, json.dumps(body))

    def cancel_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/cancel/"
        return self.make_api_request("POST", path)

    def get_order(self, order_id: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/{order_id}/"
        return self.make_api_request("GET", path)

    def get_orders(self) -> Any:
        path = "/api/v1/crypto/trading/orders/"
        return self.make_api_request("GET", path)


def main():
    api_trading_client = CryptoAPITrading()
    print(api_trading_client.get_account())

    """
    BUILD YOUR TRADING STRATEGY HERE

    order = api_trading_client.place_order(
          str(uuid.uuid4()),
          "buy",
          "market",
          "BTC-USD",
          {"asset_quantity": "0.0001"}
    )
    """


if __name__ == "__main__":
    main()
```
4. Run your script from the command line

## Authentication
Authenticated requests must include all three x-api-key, x-signature, and x-timestamp HTTP headers.

## Creating an API key
To use the Crypto Trading API, you must visit the Robinhood API Credentials Portal to create credentials. After creating credentials, you will receive the API key associated with the credential. You can modify, disable, and delete credentials you created at any time.

The API key obtained from the credentials portal will be used as the x-api-key header you will need to pass during authentication when calling our API endpoints. Additionally, you will need the public key generated in the Creating a key pair section to create your API credentials.

### Node.js
Note that you'll need to have tweetnacl and base64-js installed to run the Node.js script. You can install them with the following npm command in your terminal.

```
npm install tweetnacl base64-js
```
```javascript
const nacl = require('tweetnacl')
const base64 = require('base64-js')

// Generate an Ed25519 keypair
const keyPair = nacl.sign.keyPair()

// Convert keys to base64 strings
const private_key_base64 = base64.fromByteArray(keyPair.secretKey)
const public_key_base64 = base64.fromByteArray(keyPair.publicKey)

// Print keys in the base64 format
console.log("Private Key (Base64)")
console.log(private_key_base64)

console.log("Public Key (Base64):")
console.log(public_key_base64)
```
### Python
Note that you'll need to have pynacl installed to run the Python script. You can install them with the following pip command in your terminal.
```
pip install pynacl
```
```Python
import nacl.signing
import base64

# Generate an Ed25519 keypair
private_key = nacl.signing.SigningKey.generate()
public_key = private_key.verify_key

# Convert keys to base64 strings
private_key_base64 = base64.b64encode(private_key.encode()).decode()
public_key_base64 = base64.b64encode(public_key.encode()).decode()

# Print the keys in base64 format
print("Private Key (Base64):")
print(private_key_base64)

print("Public Key (Base64):")
print(public_key_base64)
```

## Headers and Signature

### API Key
The "x-api-key" header should contain your API key. This API key is obtained from the Robinhood API Credentials Portal when enrolling in the Robinhood Crypto API program.

*API keys issued after August 13, 2024 will be formatted as "rh-api-[uuid]." Functionality will remain the same, and older keys will keep the original formatting (no "rh-api" prepend).
Security Scheme Type: API Key
Header parameter name: x-api-key

### API Signature

Authenticated requests should be signed with the "x-signature" header, using a signature generated with the following: private key, API key, timestamp, path, method, and body. Here’s how the message signature should be defined:

```
message = f"{api_key}{current_timestamp}{path}{method}{body}"
```

*Note that for requests without a body, the body can be omitted from the message signature.

### Example Signature

The following is an example of a signature that corresponds to a cancel order request. You may use the example values below to ensure your code implementation is generating the same (x-signature) header signature value. The code snippet is for generating the signature in Python.


| Field  | Value |
| ------------- | ------------- |
| Private Key  | xQnTJVeQLmw1/Mg2YimEViSpw/SdJcgNXZ5kQkAXNPU=  |
| Public Key  |jPItx4TLjcnSUnmnXQQyAKL4eJj3+oWNNMmmm2vATqk=  |
| API Key  | rh-api-6148effc-c0b1-486c-8940-a1d099456be6  |
| Method  |  POST |
| Path  | Content Cell  |
| Public Key  | Content Cell  |
| Public Key  | Content Cell  |
| Public Key  | Content Cell  |
   
