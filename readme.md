
## Proxy settings for research cloud:
```
location / {    
  proxy_pass http://localhost:8501;
}

location ^~ /static {
    proxy_pass http://localhost:8501/static/;
}

location ^~ /healthz {
    proxy_pass http://localhost:8501/healthz;
}

location ^~ /vendor {
    proxy_pass http://localhost:8501/vendor;
}

location /_stcore/stream {
    proxy_pass http://localhost:8501/_stcore/stream;
    proxy_http_version 1.1;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $host;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 86400;
}
```