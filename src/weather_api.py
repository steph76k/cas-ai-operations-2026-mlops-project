"""Open-Meteo-Abrufe mit begrenzten Wiederholungsversuchen."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_weather(url, params):
    # Maximal drei Versuche; ungültige Anfragen (z. B. HTTP 400) nicht wiederholen.
    retries = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=False,
        raise_on_status=False,
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
