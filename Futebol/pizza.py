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

# Ordenar jogadores por tempo de tela (do maior para o menor) e selecionar top 10
player_data.sort(key=lambda x: x["screen_time"], reverse=True)
top_10_players = player_data[:10]

# Preparar dados para o gráfico
player_ids = [player["id"] for player in top_10_players]
screen_times = [player["screen_time"] for player in top_10_players]
labels = [f"{player['id']} (Entrada: {player['first_frame']}, Saída: {player['last_frame']})"
          for player in top_10_players]

# Criar o gráfico de pizza
plt.figure(figsize=(10, 8))  # Tamanho fixo para o gráfico de pizza
plt.pie(screen_times, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
plt.title("Proporção do Tempo de Tela dos Top 10 Jogadores")

# Ajustar layout para evitar sobreposição
plt.tight_layout()

# Salvar o gráfico
plt.savefig("player_screen_time_pie.png")
plt.close()  # Fechar a figura para liberar memória