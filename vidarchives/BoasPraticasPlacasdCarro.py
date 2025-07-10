from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv
from datetime import datetime
import json
import uuid
import os
import time
from dotenv import load_dotenv
import base64
from groq import Groq
from typing import Tuple, List, Dict, Optional, Set

# Configurações globais
VEHICLE_CLASSES = [2, 3, 5, 7]  # carro, moto, caminhão, ônibus
RESIZE_SCALE = 0.3
VEHICLE_CONFIDENCE = 0.6
PLATE_CONFIDENCE = 0.3
API_DELAY = 3  # segundos

class PlateDetector:
    def __init__(self, 
                 video_path: str, 
                 vehicle_model_path: str = 'yolo11s.pt',
                 plate_model_path: str = r'C:\Users\danil\Downloads\Atualizado\modelos\LicensePlateRecognition\weights\best.pt'):
        """
        Inicializa o detector de placas.
        
        Args:
            video_path: Caminho para o vídeo ou 0 para webcam
            vehicle_model_path: Caminho para o modelo YOLO de veículos
            plate_model_path: Caminho para o modelo YOLO de placas
        """
        self.video_path = video_path
        self.vehicle_model_path = vehicle_model_path
        self.plate_model_path = plate_model_path
        
        # Conjuntos e estruturas de dados
        self.unique_plates: Set[int] = set()
        self.output_json: List[Dict] = []
        
        # Inicializar componentes
        self._setup_environment()
        self._load_models()
        self._setup_trackers()
        self._setup_video_capture()
        self._setup_output_directories()
        self._initialize_output_file()
        
    def _setup_environment(self) -> None:
        """Carrega variáveis de ambiente."""
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
    def _load_models(self) -> None:
        """Carrega os modelos YOLO."""
        self.vehicle_detector = YOLO(self.vehicle_model_path)
        self.license_plate_detector = YOLO(self.plate_model_path)
        
    def _setup_trackers(self) -> None:
        """Configura os trackers e anotadores."""
        # Trackers
        self.byte_tracker = sv.ByteTrack()
        self.byte_tracker2 = sv.ByteTrack()
        
        # Anotadores
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator2 = sv.BoxAnnotator()
        self.label_annotator2 = sv.LabelAnnotator()
        
        # Mapeamento de classes
        self.class_map = {0: "Placa"}
        
    def _setup_video_capture(self) -> None:
        """Configura a captura de vídeo."""
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * RESIZE_SCALE)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_SCALE)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
    def _setup_output_directories(self) -> None:
        """Cria diretórios de saída."""
        self.output_folder = "frametoocr"
        self.output_file = "outputs/PlacaCarros.json"
        
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
    def _initialize_output_file(self) -> None:
        """Inicializa o arquivo de saída JSON."""
        try:
            with open(self.output_file, 'r') as f:
                self.output_json = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.output_json = []
        
        # Adicionar dados iniciais
        init_data = {
            "ID da Execucao do Programa": str(uuid.uuid4()),
            "Execucao do Programa as": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.output_json.append(init_data)
        self._save_json()
        
    def _save_json(self) -> None:
        """Salva os dados no arquivo JSON."""
        with open(self.output_file, 'w') as f:
            json.dump(self.output_json, f, indent=4)
            
    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Codifica uma imagem em base64.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            String da imagem codificada em base64
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def is_inside_vehicle(plate_box: Tuple[float, float, float, float], 
                         vehicle_box: Tuple[float, float, float, float]) -> bool:
        """
        Verifica se a bounding box da placa está dentro da bounding box do veículo.
        
        Args:
            plate_box: Coordenadas da placa (x1, y1, x2, y2)
            vehicle_box: Coordenadas do veículo (x1, y1, x2, y2)
            
        Returns:
            True se a placa estiver dentro do veículo
        """
        px1, py1, px2, py2 = plate_box
        vx1, vy1, vx2, vy2 = vehicle_box
        return px1 >= vx1 and py1 >= vy1 and px2 <= vx2 and py2 <= vy2
    
    def _extract_plate_text_via_api(self, image_path: str) -> str:
        """
        Extrai o texto da placa usando a API Groq.
        
        Args:
            image_path: Caminho para a imagem da placa
            
        Returns:
            Texto extraído da placa
        """
        try:
            base64_image = self.encode_image(image_path)
            print("Imagem transformada, tentando API...")
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Extract the text from the image, i only want the letters and number form the car carplate, nothing more, don't say anything else"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
            )
            
            content = chat_completion.choices[0].message.content
            print(f"Sucesso na API: {content}")
            time.sleep(API_DELAY)
            return content
            
        except Exception as e:
            print(f"Falhou na API: {str(e)}")
            time.sleep(API_DELAY)
            return "Falha na API"
    
    def _save_plate_image(self, frame: np.ndarray, plate_box: Tuple[float, float, float, float], 
                         plate_tracker_id: int) -> str:
        """
        Salva a imagem da placa recortada.
        
        Args:
            frame: Frame atual do vídeo
            plate_box: Coordenadas da placa
            plate_tracker_id: ID do tracker da placa
            
        Returns:
            Caminho do arquivo salvo
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = map(int, plate_box)
        plate_image = frame[y1:y2, x1:x2]
        
        plate_filename = os.path.join(
            self.output_folder, 
            f"plate_{plate_tracker_id}_{current_time.replace(':', '-')}.jpg"
        )
        cv2.imwrite(plate_filename, plate_image)
        return plate_filename
    
    def _process_plate_detection(self, frame: np.ndarray, plate_tracker_id: int, 
                               plate_conf: float, plate_box: Tuple[float, float, float, float],
                               vehicle_id: int) -> None:
        """
        Processa uma detecção de placa válida.
        
        Args:
            frame: Frame atual
            plate_tracker_id: ID do tracker da placa
            plate_conf: Confiança da detecção
            plate_box: Coordenadas da placa
            vehicle_id: ID do veículo correspondente
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.unique_plates.add(plate_tracker_id)
        
        # Salvar imagem da placa
        plate_filename = self._save_plate_image(frame, plate_box, plate_tracker_id)
        
        # Extrair texto via API
        extracted_text = self._extract_plate_text_via_api(plate_filename)
        
        # Criar entrada no JSON
        entry = {
            "ID do Carro": int(vehicle_id),
            "ID da Placa": int(plate_tracker_id),
            "confidence": float(plate_conf),
            "Plate Name": extracted_text,
            "exit_timestamp": current_time
        }
        self.output_json.append(entry)
        
        # Remover arquivo temporário
        try:
            os.remove(plate_filename)
            print("Arquivo removido")
        except Exception as e:
            print(f"Erro ao remover arquivo: {e}")
    
    def _find_corresponding_vehicle(self, plate_box: Tuple[float, float, float, float],
                                  vehicle_detections: sv.Detections) -> Optional[int]:
        """
        Encontra o veículo correspondente para uma placa.
        
        Args:
            plate_box: Coordenadas da placa
            vehicle_detections: Detecções de veículos
            
        Returns:
            ID do veículo correspondente ou None
        """
        for vehicle_tracker_id, vehicle_box in zip(vehicle_detections.tracker_id, 
                                                  vehicle_detections.xyxy):
            if vehicle_tracker_id is None:
                continue
                
            if self.is_inside_vehicle(plate_box, vehicle_box):
                return vehicle_tracker_id
                
        return None
    
    def _create_labels(self, detections: sv.Detections, is_vehicle: bool = True) -> List[str]:
        """
        Cria labels para as detecções.
        
        Args:
            detections: Detecções do supervision
            is_vehicle: Se True, cria labels para veículos, senão para placas
            
        Returns:
            Lista de labels
        """
        if is_vehicle:
            return [
                f"{self.vehicle_detector.names.get(class_id, f'Class_{class_id}')} (ID: {tracker_id})"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
                if tracker_id is not None
            ]
        else:
            return [
                f"{self.class_map.get(class_id, f'Class_{class_id}')} ID:({tracker_id})"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
                if tracker_id is not None
            ] if detections.class_id is not None else []
    
    def _annotate_frame(self, frame: np.ndarray, vehicle_detections: sv.Detections, 
                       plate_detections: sv.Detections) -> np.ndarray:
        """
        Anota o frame com as detecções.
        
        Args:
            frame: Frame original
            vehicle_detections: Detecções de veículos
            plate_detections: Detecções de placas
            
        Returns:
            Frame anotado
        """
        annotated_frame = frame.copy()
        
        # Labels
        vehicle_labels = self._create_labels(vehicle_detections, is_vehicle=True)
        plate_labels = self._create_labels(plate_detections, is_vehicle=False)
        
        # Anotar veículos
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=vehicle_detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=vehicle_detections, labels=vehicle_labels
        )
        
        # Anotar placas
        annotated_frame = self.box_annotator2.annotate(
            scene=annotated_frame, detections=plate_detections
        )
        annotated_frame = self.label_annotator2.annotate(
            scene=annotated_frame, detections=plate_detections, labels=plate_labels
        )
        
        # Adicionar contador
        cv2.putText(
            annotated_frame, 
            f"Placas Detectadas: {len(plate_detections)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return annotated_frame
    
    def process_video(self) -> None:
        """Processa o vídeo completo."""
        frame_count = 0
        print("Iniciando processamento do vídeo...")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (self.width, self.height))
            frame_count += 1
            
            # Fazer detecções
            vehicle_result = self.vehicle_detector(frame, conf=VEHICLE_CONFIDENCE, classes=VEHICLE_CLASSES)[0]
            plate_result = self.license_plate_detector(frame, conf=PLATE_CONFIDENCE)[0]
            
            # Converter para formato supervision
            vehicle_detections = sv.Detections.from_ultralytics(vehicle_result)
            plate_detections = sv.Detections.from_ultralytics(plate_result)
            
            # Atualizar trackers
            vehicle_detections = self.byte_tracker.update_with_detections(vehicle_detections)
            plate_detections = self.byte_tracker2.update_with_detections(plate_detections)
            
            # Processar cada placa detectada
            for plate_tracker_id, plate_conf, plate_box in zip(
                plate_detections.tracker_id, 
                plate_detections.confidence, 
                plate_detections.xyxy
            ):
                if plate_tracker_id is None or plate_tracker_id in self.unique_plates:
                    continue
                    
                # Encontrar veículo correspondente
                vehicle_id = self._find_corresponding_vehicle(plate_box, vehicle_detections)
                
                if vehicle_id is not None:
                    self._process_plate_detection(
                        frame, plate_tracker_id, plate_conf, plate_box, vehicle_id
                    )
            
            # Salvar JSON após processar todas as placas do frame
            self._save_json()
            
            # Anotar e mostrar frame
            annotated_frame = self._annotate_frame(frame, vehicle_detections, plate_detections)
            cv2.imshow('Detector de Placas', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        print(f"Processamento concluído! Total de placas únicas: {len(self.unique_plates)}")
        
    def cleanup(self) -> None:
        """Limpa recursos."""
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Função principal."""
    # Configurações
    video_path = 'videos/carsplate.mp4'
    
    # Criar e executar detector
    detector = PlateDetector(video_path)
    
    try:
        detector.process_video()
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()