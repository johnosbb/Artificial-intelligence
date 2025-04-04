## References

```
/api/embed
```

## Setting up Chroma in a container

```
docker run -d -p 8000:8000 -v ~/my-chroma-data:/chromadb/data chromadb/chroma

```

Browse to

```
http://127.0.0.1:8000/
```

## Accessing the Docker instance

```
docker exec -it charming_poitras bash
```

where

```
docker ps
CONTAINER ID   IMAGE             COMMAND                  CREATED              STATUS              PORTS                                         NAMES
fb3a008aff3c   chromadb/chroma   "/docker_entrypoint.…"   About a minute ago   Up About a minute   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   charming_poitras

```

## Docker Logs

```
docker logs <CONTAINER ID>>
```

## Setting up Ollama without a container

```
chroma run --path /mnt/500GB/ChromaDB
```

## References

- [technovangelist](https://github.com/technovangelist)
- [Video Projects](https://github.com/technovangelist/videoprojects)
