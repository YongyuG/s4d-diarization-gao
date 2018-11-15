#-*- coding:utf-8 -*-
#!/usr/bin/env python

import struct

import bson
import requests
import os


base_url = "http://127.0.0.1:5000/call"

data = ""
with open("/media/yongyu/办公/s4d/gaoS4D/023_8k.wav", "rb") as f:
    data = f.read()

req = {
    "methodName": "speaker_diarization_S4D",
    "fileMap": {"file": data}
}
data = bson.dumps(req)
result = bson.loads(
    requests.post(
        base_url,
        data=data,
        headers={
            'content-type': 'application/bson'}).content)
