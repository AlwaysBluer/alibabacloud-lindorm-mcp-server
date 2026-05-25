import json

import requests


#### LINDORM AI EMBEDDING ####
def _ensure_compatible_base_url(host: str, use_ssl: bool) -> str:
    if host.startswith(("http://", "https://")):
        base_url = host.rstrip("/")
    else:
        scheme = "https" if use_ssl else "http"
        base_url = f"{scheme}://{host}:9002"

    if base_url.endswith("/dashscope/compatible-mode/v1"):
        return base_url
    return f"{base_url}/dashscope/compatible-mode/v1"


def _post_model_request(
    host: str,
    username: str,
    password: str,
    model: str,
    data: dict,
    use_ssl: bool = False,
    verify_ssl: bool = False,
    connect_timeout: int = 60,
    read_timeout: int = 60,
):
    payload = json.dumps(data)
    url = f"{_ensure_compatible_base_url(host, use_ssl)}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "x-ld-ak": username,
        "x-ld-sk": password,
        "Accept-Encoding": "identity",
    }
    timeout = (connect_timeout, read_timeout)

    try:
        result = requests.post(url, data=payload, headers=headers, verify=verify_ssl, timeout=timeout)
        result.raise_for_status()
        embeddings = [item["embedding"] for item in result.json()["data"]]
        return 0, embeddings
    except requests.exceptions.Timeout as time_out_err:
        return -1, f"request out of time: f{time_out_err}"
    except requests.exceptions.HTTPError as http_err:
        return -1, f"HTTP error: {http_err}"
    except requests.exceptions.RequestException as err:
        return -1, f"request error happened: {err}"


def text_embedding(
    host: str,
    username: str,
    password: str,
    model: str,
    text: str,
    use_ssl: bool = False,
    verify_ssl: bool = False,
    dimensions: int | None = 1024,
):
    data = {
        "model": model,
        "input": [text],
        "encoding_format": "float",
    }
    if isinstance(dimensions, int):
        data["dimensions"] = dimensions
    return _post_model_request(
        host, username, password, model, data,
        use_ssl=use_ssl, verify_ssl=verify_ssl,
    )


def get_lindorm_search_host(instance_id: str, using_vpc: bool = False):
    """
    Get search host by instance id
    :param instance_id: Lindorm instance ID
    :param using_vpc: Boolean flag indicating whether to use VPC endpoint
    :return: Formatted search host URL
    """
    base_url = "lindorm.aliyuncs.com"
    if using_vpc:
        endpoint = "proxy-search-vpc"
    else:
        endpoint = "proxy-search-pub"
    return f"{instance_id}-{endpoint}.{base_url}"


def get_lindorm_ai_host(instance_id: str, using_vpc: bool = False):
    base_url = "lindorm.aliyuncs.com"
    if using_vpc:
        endpoint = "proxy-ai-vpc"
    else:
        endpoint = "proxy-ai-pub"
    return f"{instance_id}-{endpoint}.{base_url}"

def get_lindorm_table_host(instance_id: str, using_vpc: bool = False):
    base_url = "lindorm.aliyuncs.com"
    if using_vpc:
        endpoint = "proxy-lindorm-vpc"
    else:
        endpoint = "proxy-lindorm-pub"
    return f"{instance_id}-{endpoint}.{base_url}"


def str_to_bool(value):
    return value.lower() in ('true', '1', 'yes', 'on', 't')

def simplify_mappings(mappings, index_name):
    if not mappings or index_name not in mappings:
        return None

    properties = mappings[index_name]['mappings'].get('properties', {})
    simplified = {}

    for field, details in properties.items():
        if 'type' in details:
            simplified[field] = details['type']
        elif 'properties' in details:
            simplified[field] = 'object'
        else:
            simplified[field] = 'unknown'

    return simplified
