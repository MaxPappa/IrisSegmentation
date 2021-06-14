from pathlib import Path
import zipfile
import os
from urllib.request import urlopen
from io import BytesIO


def downloadDataset(p: Path, url: str):
    print(
        f"### Starting to download the dataset and extracting it into {p.name} folder ###"
    )
    os.mkdir(p.name)
    resp = urlopen(url)
    zip_obj = zipfile.ZipFile(BytesIO(resp.read()))  # maybe just try to handle errors
    print("### Download completed. Wait till the extraction has been completed ###")
    zip_obj.extractall(f"./{p.name}")
    print("### Download and extraction completed ###")
