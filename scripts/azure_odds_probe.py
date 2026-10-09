from __future__ import annotations
import requests, json
from pathlib import Path
OUT=Path('results/azure_odds_probe'); OUT.mkdir(parents=True,exist_ok=True)
base='https://metricstrackersgbsv.blob.core.windows.net/ncaam-historical-raw'
url=base+'?restype=container&comp=list&prefix=odds/raw/archive/'
r=requests.get(url,timeout=30)
out={'list_status':r.status_code,'headers':dict(r.headers),'body_prefix':r.text[:5000]}
# Also test likely root raw folder listing and a bare container HEAD/GET
r2=requests.get(base+'?restype=container&comp=list&prefix=odds/raw/',timeout=30)
out['raw_list_status']=r2.status_code; out['raw_body_prefix']=r2.text[:5000]
(OUT/'summary.json').write_text(json.dumps(out,indent=2,default=str))
print('AZURE_PROBE'); print(json.dumps(out,indent=2,default=str))
