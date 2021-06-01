from pathlib import Path
import zipfile, os
from urllib.request import urlopen
from io import BytesIO

def downloadDataset(p: Path, url: str):
    print(f"### Starting to download the dataset and extracting it into {p.name} folder ###")
    os.mkdir(p.name)
    resp = urlopen(url)
    zip = zipfile.ZipFile(BytesIO(resp.read())) # maybe just try to handle errors
    print(f"### Download completed. Wait till the extraction has been completed ###")
    zip.extractall(f'./{p.name}')
    print(f"### Download and extraction completed ###")