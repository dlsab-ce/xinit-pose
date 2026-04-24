import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HOST = "0.0.0.0"
#PORT = 9092
#BASE_DIR = "./output"
BASE_DIR = "/workspace/output"
PORT = 8080

# Crea la cartella output se non esiste
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Array globale per memorizzare le immagini
# Struttura: lista di tuple (output_filename, timestamp, image_bytes)
images_cache = []

MAX_CACHE_SIZE = 10

class LoggerHandler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _add_to_cache(self, output_filename: str, timestamp: str, image_bytes: bytes) -> None:
        """Aggiungi una tupla alla cache mantenendola ordinata (più recente in posizione 0)
        e limitandola a MAX_CACHE_SIZE elementi."""
        global images_cache
        # Inserisci in posizione 0 (più recente viene per primo)
        images_cache.insert(0, (output_filename, timestamp, image_bytes))
        # Rimuovi elementi più vecchi se supera MAX_CACHE_SIZE
        if len(images_cache) > MAX_CACHE_SIZE:
            images_cache = images_cache[:MAX_CACHE_SIZE]

    def _get_images_list(self) -> list:
        """Ritorna la lista dei nomi delle immagini dalla cache, ordinate per data decrescente (più recente primo)"""
        # Gli elementi sono già ordinati con i più recenti in posizione 0
        return [filename for filename, _, _ in images_cache]

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/":
            # Serve la home page
            try:
                html_path = os.path.join(os.path.dirname(__file__), "home.html")
                with open(html_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                self._send_html(200, html)
            except FileNotFoundError:
                self._send_json(500, {"error": "home.html not found"})

        elif self.path == "/gallery":
            # Serve la pagina della galleria
            try:
                html_path = os.path.join(os.path.dirname(__file__), "gallery.html")
                with open(html_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                self._send_html(200, html)
            except FileNotFoundError:
                self._send_json(500, {"error": "gallery.html not found"})

        elif self.path == "/api/latest-image":
            # API endpoint che ritorna solo l'ultima immagine
            images = self._get_images_list()
            if images:
                self._send_json(200, {"image": images[0], "total": len(images)})
            else:
                self._send_json(200, {"image": None, "total": 0})

        elif self.path.startswith("/api/images"):
            # API endpoint che ritorna la lista di immagini
            images = self._get_images_list()
            self._send_json(200, {"images": images})

        elif self.path.startswith("/images/"):
            # Serve le immagini dalla cache in memoria
            # Estrai il nome del file rimuovendo i parametri querystring
            image_name = self.path.split("/images/")[1].split('?')[0]
            
            # Cerca il file nella cache
            image_data = None
            for filename, _, image_bytes in images_cache:
                if filename == image_name:
                    image_data = image_bytes
                    break
            
            if image_data:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(image_data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(image_data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self._send_json(404, {"error": "not found"})


    def do_POST(self):
        """
        Handle POST requests to the server.
        Expects: JSON with image bytes in request body
        Query params:
            - store_files: Whether to save images to disk (default: False)
        
        Logica:
        1. Converte il JSON ricevuto in immagine
        2. Registra i bytes dell'immagine nella mappa globale
        3. Scrive su disco SOLO se store_files è True
        """
        global images_cache
        
        # Estrai i parametri dalla query string
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        store_files = query_params.get('store_files', ['false'])[0].lower() == 'true'
        
        # Verifica il path principale (senza query string)
        if parsed_url.path != "/image":
            self._send_json(404, {"error": "not found"})
            return
        try:    
            content_length = int(self.headers.get("Content-Length", 0))
            json_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            image_bytes_data = json_data['outputs'][0]['data']

            # Genera timestamp con millisecondi usando datetime.now() moderno
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:-3]
            output_filename = f"image_{timestamp}.jpg"

            # Converti bytes a immagine PIL per validazione
            image_bytes_clean = bytes(image_bytes_data)
            image = Image.open(BytesIO(image_bytes_clean)).convert('RGB')
            
            # Aggiungi la tupla alla cache tramite il metodo dedicato
            self._add_to_cache(output_filename, timestamp, image_bytes_clean)
            
            # Scrivi su disco SOLO se store_files è True
            if store_files:
                full_path = os.path.join(BASE_DIR, output_filename)
                image.save(full_path)
            
            self._send_json(200, {
                "status": "ok", 
                "filename": output_filename,
                "timestamp": timestamp,
                "store_files": store_files,
                "cached": True,
                "cache_size": len(images_cache)
            })
        except (json.JSONDecodeError, ValueError) as ex:
            self._send_json(400, {"error": f"Invalid JSON: {ex}"})
        except Exception as ex:
            self._send_json(500, {"error": str(ex)})
        

def main() -> None:
    server = HTTPServer((HOST, PORT), LoggerHandler)
    print(f"Xinet file logger server listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()