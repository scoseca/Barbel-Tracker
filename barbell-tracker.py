import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
from collections import defaultdict
import torch

print(f"CUDA Disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Numero di GPU: {torch.cuda.device_count()}")

class BarbellTracker:
    def __init__(self, video_path, output_path, confidence=0.5, class_id=None, device='cuda'):
        self.video_path = video_path
        self.output_path = output_path
        self.confidence = confidence
        self.device = device


        if self.device == 'cuda' and not torch.cuda.is_available():
            print("AVVISO: CUDA non è disponibile. Usando CPU.")
            self.device = 'cpu'
        
        print(f"Usando dispositivo: {self.device}")
        
        # Inizializzazione del modello YOLOv8 con il dispositivo specificato
        self.model = YOLO('yolov8n.pt').to(self.device)
    
        # Inizializzazione del modello YOLOv8
        # self.model = YOLO('yolov8l.pt').to('cuda')  # Usa la GPU
        # self.model = YOLO('yolov8l.pt')  # Modello piccolo, carica yolov8m.pt o yolov8l.pt per maggiore precisione
        
        # Inizializzazione del tracker DeepSORT
        if self.device == 'cuda':
            self.tracker = DeepSort(
                max_age=30,
                nn_budget=100,
                embedder="mobilenet",  # Modello più leggero, adatto per GPU
                embedder_gpu=True      # Usa GPU per l'embedding
            )
        else:
            self.tracker = DeepSort(max_age=30)
        
        # Classe del bilanciere
        self.barbell_class_id = class_id
        
        # Dizionario per memorizzare le traiettorie
        self.trajectories = defaultdict(list)
        
    def getVideoProperties(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {self.video_path}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, fps

    def process_video(self):
        # Apertura del video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {self.video_path}")
            return
        
        # Ottenere le proprietà del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Configurazione dell'output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # Usa il dispositivo specificato
            if self.barbell_class_id is not None:
                results = self.model(frame, classes=[self.barbell_class_id], device=self.device)
            else:
                results = self.model(frame, device=self.device)

            # Esecuzione dell'inferenza con YOLOv8
            # results = self.model(frame)
            
            # Estrazione delle bounding box per i bilancieri
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Verifica se l'oggetto è un bilanciere (o simile) e se la confidenza è sufficiente
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > self.confidence:
                        # Per ora, consideramo tutte le classi come potenziali bilancieri
                        # In un'implementazione reale, dovresti filtrare solo per bilancieri
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Aggiungi rilevamento per DeepSORT
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
            
            # Aggiornamento del tracker DeepSORT
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Aggiornamento delle traiettorie e disegno su frame
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # Calcolo del punto centrale dell'oggetto
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Aggiunta del punto alla traiettoria
                self.trajectories[track_id].append((center_x, center_y))
                
                # Disegno della bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Disegno della traiettoria
                points = self.trajectories[track_id]
                for i in range(1, len(points)):
                    # Colore che cambia gradualmente lungo la traiettoria
                    color_intensity = min(255, i * 5)
                    cv2.line(frame, points[i-1], points[i], (0, 0, color_intensity), 2)
            
            # Scrittura del frame nel video di output
            out.write(frame)
            
            # Visualizzazione 
            cv2.imshow('Barbell Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Pulizia
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video elaborato salvato come {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traccia la traiettoria di un bilanciere in un video.')
    parser.add_argument('--input', required=True, help='Percorso del video di input')
    parser.add_argument('--output', required=True, help='Percorso del video di output')
    parser.add_argument('--confidence', type=float, default=0.5, help='Soglia di confidenza per il rilevamento (default: 0.5)')
    parser.add_argument('--class_id', type=int, help='ID della classe da rilevare (se non specificato, rileva tutte le classi)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Dispositivo da utilizzare (default: cuda)')
    
    args = parser.parse_args()
    
    tracker = BarbellTracker(args.input, args.output, args.confidence, args.class_id, args.device)
    print(tracker.getVideoProperties())
    # tracker.model.train(data="coco.yaml", epochs=100, imgsz=640) 
    tracker.process_video()