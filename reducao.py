import cv2

# Carregar a imagem
imagem = cv2.imread('Lenna.png')

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem!")
    exit()

# 1. Converter para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# 2. Converter para binário (preto e branco) usando o método de Otsu
_, imagem_binaria = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Salvar ou exibir as imagens
cv2.imwrite('imagem_cinza.jpg', imagem_cinza)  # Salvar imagem em cinza
cv2.imwrite('imagem_binaria.jpg', imagem_binaria)  # Salvar imagem binária

# Exibir as imagens (opcional)
cv2.imshow('Imagem Normal', imagem)
cv2.imshow('Imagem em Cinza', imagem_cinza)
cv2.imshow('Imagem Binaria', imagem_binaria)
cv2.waitKey(0)  # Aguardar pressionar uma tecla
cv2.destroyAllWindows()  # Fechar as janelas