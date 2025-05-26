import cv2
import os

def video_to_frames(video_path, interval=60, output_format='jpg', quality=100, resize_factor=None):
    """
    Extrai frames de um vídeo e os salva com qualidade controlada.
    
    Args:
        video_path (str): Caminho do vídeo.
        interval (int): Intervalo de frames a serem salvos (padrão: 40).
        output_format (str): Formato de saída ('jpg' ou 'png').
        quality (int): Qualidade JPEG (0-100, padrão: 100). Ignorado para PNG.
        resize_factor (float): Fator de redimensionamento (ex.: 0.5 para metade da resolução).
    """
    # Cria a pasta 'results' se ela não existir
    output_folder = 'results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Abre o vídeo
    video = cv2.VideoCapture(video_path)
    
    # Verifica se o vídeo foi aberto corretamente
    if not video.isOpened():
        print("Erro ao abrir o vídeo!")
        return

    # Obtém informações do vídeo
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Vídeo: {total_frames} frames, {fps:.2f} FPS")

    # Contadores
    frame_count = 0
    saved_count = 0
    success = True

    # Configura parâmetros de qualidade para JPEG
    if output_format.lower() == 'jpg':
        save_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        file_extension = '.jpg'
    else:  # PNG
        save_params = []
        file_extension = '.png'

    # Lê frame por frame
    while success:
        success, frame = video.read()
        
        if success:
            # Aplica redimensionamento, se especificado
            if resize_factor is not None and resize_factor != 1.0:
                width = int(frame.shape[1] * resize_factor)
                height = int(frame.shape[0] * resize_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Salva frame no intervalo especificado
            if frame_count % interval == 0:
                frame_name = f"{output_folder}/frame{saved_count:04d}{file_extension}"
                cv2.imwrite(frame_name, frame, save_params)
                print(f"Frame {frame_count} salvo como {frame_name}")
                saved_count += 1
            frame_count += 1

    # Libera o vídeo
    video.release()
    print(f"Concluído! {saved_count} frames salvos na pasta '{output_folder}' "
          f"de um total de {frame_count} frames")

if __name__ == "__main__":
    # Exemplo de uso
    video_path = "cafesplantaszapb.mp4"
    # Configurações para máxima qualidade
    video_to_frames(
        video_path,
        interval=15,
        output_format='png',  # Use 'jpg' para arquivos menores ou 'png' para qualidade sem perdas
        quality=100,          # Máxima qualidade para JPEG (ignorado para PNG)
        resize_factor=None    # Não redimensiona (use, por exemplo, 0.5 para reduzir à metade)
    )