import numpy as np
from PIL import Image
import os

class SistemaImagem:
    def __init__(self):
        self.imagem = None
        self.caminho_imagem = None

    def abrir_imagem(self, caminho):
        extensoes_suportadas = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        if not caminho.lower().endswith(extensoes_suportadas):
            print(f"[ERRO] Formato não suportado: {caminho}")
            return
        if not os.path.exists(caminho):
            print(f"[ERRO] Arquivo de imagem não encontrado: {caminho}")
            return
        try:
            img = Image.open(caminho)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            self.imagem = img
            self.caminho_imagem = caminho
            print(f"[OK] Imagem carregada: {caminho}")
        except Exception as e:
            print(f"[ERRO] Falha ao abrir a imagem: {e}")

    def exibir_imagem(self):
        if self.imagem:
            self.imagem.show()
        else:
            print("[INFO] Nenhuma imagem carregada.")

    def salvar_imagem(self, novo_caminho):
        if self.imagem:
            try:
                self.imagem.save(novo_caminho)
                print(f"[OK] Imagem salva em: {novo_caminho}")
            except Exception as e:
                print(f"[ERRO] Falha ao salvar a imagem: {e}")
        else:
            print("[INFO] Nenhuma imagem para salvar.")

# ==============================================
# FUNÇÕES DE FUNCIONALIDADES SEPARADAS
# ==============================================

def expandir_histograma(canal):
    min_val = np.min(canal)
    max_val = np.max(canal)
    if max_val - min_val == 0:
        return np.zeros_like(canal)
    return (canal - min_val) * 255.0 / (max_val - min_val)

# Funcionalidade 1
def funcionalidade_1(imagem: Image.Image, caminho_filtro: str) -> Image.Image:
    try:
        with open(caminho_filtro, 'r') as f:
            tipo_filtro = f.readline().strip().lower()
            tamanho = int(f.readline().strip())
            mascara = [list(map(float, f.readline().strip().split())) for _ in range(tamanho)]
            kernel = np.array(mascara)
    except Exception as e:
        print(f"[ERRO] Erro ao ler o filtro: {e}")
        return imagem

    img_array = np.array(imagem)
    resultado = np.zeros_like(img_array)

    for canal in range(3):  # R, G, B
        resultado[:, :, canal] = correlacao2d(img_array[:, :, canal], kernel, tipo_filtro)

    return Image.fromarray(np.uint8(np.clip(resultado, 0, 255)))

def correlacao2d(canal, kernel, tipo_filtro):
    altura, largura = canal.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    output = np.zeros_like(canal, dtype=float)

    for i in range(ph, altura - (kh - ph - 1)):
        for j in range(pw, largura - (kw - pw - 1)):
            regiao = canal[i - ph:i + (kh - ph), j - pw:j + (kw - pw)]
            if regiao.shape == kernel.shape:
                valor = np.sum(regiao * kernel)
                output[i, j] = valor

    if "sobel" in tipo_filtro:
        output = np.abs(output)
        output = expandir_histograma(output)

    return output

# Funcionalidade 2
def funcionalidade_2(imagem: Image.Image, caminho_filtro: str) -> Image.Image:
    try:
        with open(caminho_filtro, 'r') as f:
            tipo_filtro = f.readline().strip().lower()
            bias = int(f.readline().strip())
            passo = int(f.readline().strip())
            ativacao = f.readline().strip().lower()
            tamanho = int(f.readline().strip())
            mascara = [list(map(float, f.readline().strip().split())) for _ in range(tamanho)]
            kernel = np.array(mascara)

            if "sobel" not in tipo_filtro and np.sum(kernel) != 0:
                kernel = kernel / np.sum(kernel)

    except Exception as e:
        print(f"[ERRO] Erro ao ler o filtro: {e}")
        return imagem

    img_array = np.array(imagem)
    resultado = np.zeros_like(img_array)

    for canal in range(3):  # R, G, B
        resultado[:, :, canal] = correlacao_com_parametros(
            img_array[:, :, canal], kernel, tipo_filtro, bias, passo, ativacao)

    return Image.fromarray(np.uint8(np.clip(resultado, 0, 255)))

def funcionalidade_3(imagem: Image.Image, caminho_filtro: str) -> Image.Image:
    try:
        with open(caminho_filtro, 'r') as f:
            tipo_filtro = f.readline().strip().lower()
            altura = int(f.readline().strip())
            largura = int(f.readline().strip())
            profundidade = int(f.readline().strip())  # deve ser 3
            valores = []
            for _ in range(altura * profundidade):
                linha = list(map(float, f.readline().strip().split()))
                valores.append(linha)
            kernel = np.array(valores).reshape((profundidade, altura, largura)).transpose(1, 2, 0)

    except Exception as e:
        print(f"[ERRO] Erro ao ler o filtro 3D: {e}")
        return imagem

    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
    img_array = np.array(imagem).astype(float)

    h, w, c = img_array.shape
    kh, kw, kc = kernel.shape
    ph, pw = kh // 2, kw // 2

    output = np.zeros((h, w), dtype=float)

    for i in range(ph, h - (kh - ph - 1)):
        for j in range(pw, w - (kw - pw - 1)):
            bloco = img_array[i - ph:i + (kh - ph), j - pw:j + (kw - pw), :]
            if bloco.shape == kernel.shape:
                valor = np.sum(bloco * kernel)
                output[i, j] = valor

    output = np.abs(output)
    output = expandir_histograma(output)
    output_rgb = np.stack([output] * 3, axis=-1)

    return Image.fromarray(np.uint8(np.clip(output_rgb, 0, 255)))


def funcionalidade_4(imagem: Image.Image, metodo: str = "g") -> Image.Image:
    if imagem.mode != "RGB":
        imagem = imagem.convert("RGB")
    img_array = np.array(imagem).astype(float)

    if metodo.lower() == "g":
        g = img_array[:, :, 1]
        cinza = np.stack([g, g, g], axis=-1)
    elif metodo.lower() == "y":
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cinza = np.stack([y, y, y], axis=-1)
    else:
        print("[ERRO] Método inválido. Use 'g' ou 'y'.")
        return imagem

    return Image.fromarray(np.uint8(np.clip(cinza, 0, 255)))



def correlacao_com_parametros(canal, kernel, tipo_filtro, bias, passo, ativacao):
    altura, largura = canal.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    output = np.zeros_like(canal, dtype=float)

    for i in range(ph, altura - (kh - ph - 1), max(passo, 1)):
        for j in range(pw, largura - (kw - pw - 1), max(passo, 1)):
            regiao = canal[i - ph:i + (kh - ph), j - pw:j + (kw - pw)]
            if regiao.shape == kernel.shape:
                valor = np.sum(regiao * kernel) + bias
                if ativacao == "relu":
                    valor = max(0, valor)
                output[i, j] = valor

    if "sobel" in tipo_filtro:
        output = np.abs(output)
        output = expandir_histograma(output)

    return output

if __name__ == "__main__":
    sistema = SistemaImagem()
    sistema.abrir_imagem("Testes/testpat.1k.color2.tif")

    # Funcionalidade 1
    imagem1 = funcionalidade_1(sistema.imagem, "Testes/sobelHorizontal.txt")
    imagem1.show()
    imagem1.save("resultadoSobelHorizontal_func1.jpg")

    # Funcionalidade 2
    imagem2 = funcionalidade_2(sistema.imagem, "Testes/sobelFunc2.txt")
    imagem2.show()
    imagem2.save("resultadoSobelHorizontal_func2.jpg")

    # Funcionalidade 3
    imagem3 = funcionalidade_3(sistema.imagem, "Testes/box3d.txt")
    imagem3.show()
    imagem3.save("resultadoBox3d_func3.jpg")

    # Funcionalidade 4 (a) - Replicando canal G
    imagem4a = funcionalidade_4(sistema.imagem, metodo="g")
    imagem4a.show()
    imagem4a.save("resultadoCinzaG_func4.jpg")

    # Funcionalidade 4 (b) - Replicando componente Y (YIQ)
    imagem4b = funcionalidade_4(sistema.imagem, metodo="y")
    imagem4b.show()
    imagem4b.save("resultadoCinzaYIQ_func4.jpg")