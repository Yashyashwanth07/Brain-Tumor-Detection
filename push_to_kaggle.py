import json
import urllib.request

# Load the refined notebook content and strip cached execution outputs
with open('swin-tiny (1).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb.get('cells', []):
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None

nb_content = json.dumps(nb, indent=1)
print(f"Cleaned source payload size: {len(nb_content.encode('utf-8')) / 1024:.1f} KB")

# Prepare request payload for save_notebook
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "save_notebook",
        "arguments": {
            "request": {
                "id": 135997190,
                "slug": "yashwanthboggavarapu/swin-tiny",
                "newTitle": "swin_tiny",
                "kernelType": "notebook",
                "language": "python",
                "isPrivate": True,
                "enableGpu": True,
                "enableInternet": True,
                "kernelExecutionType": "SaveAndRunAll",
                "datasetDataSources": ["yashwanthboggavarapu/braintumor"],
                "text": nb_content
            }
        }
    }
}

req_data = json.dumps(payload).encode('utf-8')
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "Authorization": "Bearer KGAT_fe5f806c2ea68a95f38e54e53c7f4712"
}

req = urllib.request.Request("https://www.kaggle.com/mcp", data=req_data, headers=headers, method="POST")

print("Pushing refined notebook to Kaggle on GPU (SaveAndRunAll)...")
try:
    with urllib.request.urlopen(req) as response:
        res_bytes = response.read().decode('utf-8')
        print("Response received:")
        for line in res_bytes.split("\n"):
            if line.startswith("data: "):
                data_json = json.loads(line[6:])
                print(json.dumps(data_json, indent=2)[:1000])
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} - {e.read().decode('utf-8')}")
except Exception as e:
    print(f"Error: {e}")
