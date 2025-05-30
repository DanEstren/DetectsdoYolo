import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Ler o arquivo JSON
with open("outputs/tracking_results.json", "r") as f:
    data = json.load(f)

# Extrair informações dos tracks
tracks = data["tracks"]
player_data = []
for player_id, info in tracks.items():
    first_frame = info["first_frame"]
    last_frame = info["last_frame"]
    screen_time = last_frame - first_frame
    player_data.append({
        "id": player_id,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "screen_time": screen_time
    })

# Ordenar jogadores por tempo de tela (do maior para o menor)
player_data.sort(key=lambda x: x["screen_time"], reverse=True)

# Preparar dados para o gráfico
player_ids = [player["id"] for player in player_data]
screen_times = [player["screen_time"] for player in player_data]
first_frames = [player["first_frame"] for player in player_data]
last_frames = [player["last_frame"] for player in player_data]

# Criar o gráfico
plt.figure(figsize=(10, max(6, len(player_ids) * 0.4)))  # Ajustar altura conforme número de jogadores
bars = plt.barh(player_ids, screen_times, color="skyblue")

# Adicionar anotações com frames de entrada e saída
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width / 2, bar.get_y() + bar.get_height() / 2,
             f"Entrada: {first_frames[i]}\nSaída: {last_frames[i]}",
             ha="center", va="center", fontsize=10, color="black")

# Configurar o gráfico
plt.xlabel("Tempo de Tela (frames)")
plt.ylabel("ID do Jogador")
plt.title("Jogadores com Maior Tempo de Tela")
plt.grid(True, axis="x", linestyle="--", alpha=0.7)

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Salvar o gráfico
plt.savefig("graficos/player_screen_time.png")
plt.close()  # Fechar a figura para liberar memória