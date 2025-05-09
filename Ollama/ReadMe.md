# Ollama

## Setting up Ollama

Create an Ollama Service

```
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
Environment="OLLAMA_HOST=0.0.0.0"
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=<your path>

[Install]
WantedBy=default.target
```

```
sudo systemctl enable ollama
sudo systemctl daemon-reload
systemctl restart  ollama
```

## Setting up Open Web UI

```
docker run -d -p 3000:8080 -e OLLAMA_BASE_URL=http://192.168.1.191:11434 -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

```
sudo ufw allow 11434/tcp
```

```
docker rm -f open-webui
```

Find where Docker stores the volume

```
docker volume inspect open-webui
```

Go to that directory, for example:

```
cd /var/lib/docker/volumes/open-webui/_data
```

For example, uploaded documents are stored in

```
/var/lib/docker/volumes/open-webui/_data/uploads
```

## Evaluating RAG

### Quick Start

I have put some evaluation code in the RAG directory.

- Install Ollama `curl -fsSL https://ollama.com/install.sh | sh`
- Install a model `ollama pull llama2`
- Then install chromadb `pip install chromadb`
- Then run the server creating a default database: `chroma run --path /mnt/500GB/ChromaDB`
- import_mkdocs.py creates the database and assumes your docs are in `/mnt/500gb/docs`
- search.py allows you to interogate the resulting embeddings

### Setting up Chroma in a container

```
docker run -d -p 8000:8000 -v ~/my-chroma-data:/chromadb/data chromadb/chroma

```

Browse to

```
http://127.0.0.1:8000/
```

### Accessing the Docker instance

```
docker exec -it charming_poitras bash
```

where

```
docker ps
CONTAINER ID   IMAGE             COMMAND                  CREATED              STATUS              PORTS                                         NAMES
fb3a008aff3c   chromadb/chroma   "/docker_entrypoint.…"   About a minute ago   Up About a minute   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   charming_poitras

```

### Docker Logs

```
docker logs <CONTAINER ID>>
```

### Setting up ChromaDB without a container

```
chroma run --path /mnt/500GB/ChromaDB
```

### Configuring Chroma as a Service

```
sudo nano /etc/systemd/system/chromadb.service
```

Create a script to launch Chroma

```
#!/bin/bash
exec /home/$USER/anaconda3/bin/chroma run --path /mnt/500GB/ChromaDB
```

```
sudo chmod +x /usr/local/bin/start-chromadb.sh
```

Enable and start the service

```
 sudo systemctl enable chromadb
 sudo systemctl start chromadb
 sudo systemctl status chromadb
```

```
[Unit]
Description=ChromaDB Vector Database Service
After=network.target

[Service]
Type=simple
User=johnos
WorkingDirectory=/home/<you name>
ExecStart=/usr/local/bin/start-chromadb.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Checking the collections

![image](https://github.com/user-attachments/assets/9fdef661-d7cb-4ba5-96d5-ce76de3b365d)

### Getting the default database

```
curl -X 'GET'   'http://localhost:8000/api/v2/tenants/default_tenant/databases'   -H 'accept: application/json'
```

returns

```json
[
  {
    "id": "00000000-0000-0000-0000-000000000000",
    "name": "default_database",
    "tenant": "default_tenant"
  }
]
```

### Listing collections

```
curl -X 'GET' \
  'http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections' \
  -H 'accept: application/json'
```

```json
[
  {
    "id": "358e6fea-3359-439c-a48d-d53a4f5d04c9",
    "name": "my_test_collection",
    "metadata": null,
    "dimension": null,
    "tenant": "default_tenant",
    "database": "default_database",
    "log_position": 0,
    "version": 0,
    "configuration_json": {
      "_type": "CollectionConfigurationInternal",
      "hnsw_configuration": {
        "M": 16,
        "_type": "HNSWConfigurationInternal",
        "batch_size": 100,
        "ef_construction": 100,
        "ef_search": 100,
        "num_threads": 16,
        "resize_factor": 1.2,
        "space": "l2",
        "sync_threshold": 1000
      }
    }
  },
  {
    "id": "b3dbf6df-ddf0-4fde-86fb-63d73de6d60c",
    "name": "buildragwithpython",
    "metadata": { "hnsw:space": "cosine" },
    "dimension": 768,
    "tenant": "default_tenant",
    "database": "default_database",
    "log_position": 0,
    "version": 0,
    "configuration_json": {
      "_type": "CollectionConfigurationInternal",
      "hnsw_configuration": {
        "M": 16,
        "_type": "HNSWConfigurationInternal",
        "batch_size": 100,
        "ef_construction": 100,
        "ef_search": 100,
        "num_threads": 16,
        "resize_factor": 1.2,
        "space": "l2",
        "sync_threshold": 1000
      }
    }
  }
]
```

## Environment Variables

- OLLAMA_DEBUG=1 ollama serve
- OLLAMA_HOST
- OLLAMA_KEEP_ALIVE # time for model to remain in memory
- OLLAMA_MODELS
- OLLAMA_MAX_LOADED_MODELS # 3 times the number of GPUs
- OLLAMA_NUM_PARALLEL # defaults 4 or 1
- OLLAMA_MAX_QUEUE # defaults 512

## Linux

- sudo systemctl edit ollala.service

```
  [service]
  Environment="OLLAMA_HOST=127.0.0.1"
```

- sudo systemctl daemon-relaod
- sudo systemctl restart ollama

## Search Integration

- searchng
-

## References

- [technovangelist](https://github.com/technovangelist)
- [Video Projects](https://github.com/technovangelist/videoprojects)
- [sbert - embedding models](https://sbert.net/)

```
/api/embed
```
