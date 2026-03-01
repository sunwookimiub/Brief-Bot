import urllib.request, json, urllib.error

req = urllib.request.Request(
    'http://localhost:8080/ingest',
    data=json.dumps({'gcs_prefix': 'manuals/test/', 'doc_version': 'v1'}).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}:")
    print(e.read().decode())
