"""Local web inference. Run: python web_demo.py --model GoEmotions_BERT.goemotions"""
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import sys
import webbrowser

try:
    from portable import PortablePredictor
except ImportError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.models.portable import PortablePredictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--port',type=int,default=7870)
    p.add_argument('--open',action='store_true')
    a=p.parse_args()
    predictor=PortablePredictor(a.model,a.device)
    html=(Path(__file__).parent/'index.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,payload,kind='application/json; charset=utf-8'):
            if not isinstance(payload,bytes):payload=json.dumps(payload,ensure_ascii=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/':self.send(200,html,'text/html; charset=utf-8')
            elif self.path=='/api/info':self.send(200,{k:predictor.manifest[k] for k in ['architecture','seed','max_length','label_names']})
            else:self.send(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/predict':return self.send(404,{'error':'Not found'})
            if self.headers.get('Origin') not in (None,f'http://127.0.0.1:{a.port}',f'http://localhost:{a.port}'):
                return self.send(403,{'error':'Local requests only'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=100000:raise ValueError('Request too large or empty')
                data=json.loads(self.rfile.read(size))
                self.send(200,predictor.predict(data.get('text'),data.get('mode','tuned')))
            except (ValueError,TypeError,AttributeError) as error:self.send(400,{'error':str(error)})
    server=HTTPServer(('127.0.0.1',a.port),Handler)
    url=f'http://127.0.0.1:{a.port}'
    print(url,flush=True)
    if a.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
