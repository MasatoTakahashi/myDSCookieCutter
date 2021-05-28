from string import Template
from typing import Dict


def embed_params(template_str: str, dict_embedding_pair: Dict) -> str:
    ret = Template(template_str).safe_substitute(dict_embedding_pair)
    return(ret)
