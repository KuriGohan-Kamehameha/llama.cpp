#!/usr/bin/env python3
"""Disposable llama-server drain integration, CPU/tiny model only."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SERVER, MODEL = sys.argv[1:]
PORT = 19083

def request(conn, method, path, body=None):
    data = json.dumps(body) if body is not None else None
    conn.request(method, path, data, {'Content-Type': 'application/json'})
    response = conn.getresponse()
    return response, response.read()

def start(marker, log):
    env = dict(os.environ, LLAMA_DRAIN_MARKER=str(marker))
    return subprocess.Popen([SERVER, '-m', MODEL, '--host', '127.0.0.1', '--port', str(PORT),
                             '-t', '1', '-c', '1024', '-np', '2', '--no-ui', '-ngl', '0'],
                            env=env, stdout=log, stderr=subprocess.STDOUT)

def ready(process):
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError('server exited during startup')
        try:
            c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=1)
            r, _ = request(c, 'GET', '/health'); c.close()
            if r.status == 200:
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(.1)
    raise RuntimeError('startup timed out')

def state():
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=5)
    r, data = request(c, 'GET', '/_internal/drain'); c.close()
    if r.status != 200:
        raise RuntimeError('drain status failed')
    return json.loads(data)

def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)

with tempfile.TemporaryDirectory(prefix='llama-drain-http-') as tmp:
    marker = Path(tmp) / 'hold'
    process = None
    with open(Path(tmp) / 'server.log', 'wb') as log:
        try:
            process = start(marker, log); ready(process)
            assert state() == {'configured': True, 'draining': False, 'active': 0}
            keep = http.client.HTTPConnection('127.0.0.1', PORT, timeout=5)
            assert request(keep, 'GET', '/health')[0].status == 200
            stream = http.client.HTTPConnection('127.0.0.1', PORT, timeout=60)
            stream.request('POST', '/completion', json.dumps({'prompt':'Once upon a time', 'n_predict':700,
                           'ignore_eos':True, 'stream':True}), {'Content-Type':'application/json'})
            res = stream.getresponse(); assert res.status == 200
            assert res.readline().startswith(b'data:')
            marker.write_text('maintenance\n')
            s = state(); assert s['draining'] and s['active'] >= 1, s
            r, _ = request(keep, 'POST', '/completion', {'prompt':'blocked','n_predict':4})
            assert r.status == 503, r.status
            body = res.read(); assert b'[DONE]' in body or b'"stop":true' in body
            stream.close()
            for _ in range(100):
                if state()['active'] == 0: break
                time.sleep(.1)
            assert state()['active'] == 0
            keep.close(); stop(process); process = start(marker, log); ready(process)
            assert state()['draining'] and state()['active'] == 0
            c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=5)
            assert request(c, 'POST', '/completion', {'prompt':'blocked','n_predict':4})[0].status == 503
            marker.unlink()
            assert request(c, 'POST', '/completion', {'prompt':'test','n_predict':4})[0].status == 200
            c.close(); print('HTTP_STREAM_KEEPALIVE_DRAIN_RESTART_OK')
        except Exception:
            log.flush()
            print((Path(tmp) / 'server.log').read_text(errors='replace')[-5000:], file=sys.stderr)
            raise
        finally:
            stop(process)
