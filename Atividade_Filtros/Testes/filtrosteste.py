import numpy as np 
from PIL import Image 
import os

class SistemaImagem:
    def __init__(self):
        self.imagem = None
        self.caminho_imagem = None

    def abrir_imagem(self, caminho):
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

    def aplicar_filtro_txt(self, caminho_filtro):
        if self.imagem is None:
            print("[INFO] Nenhuma imagem carregada.")
            return

        try:
            with open(caminho_filtro, 'r') as f:
                tipo_filtro = f.readline().strip().lower()
                tamanho = int(f.readline().strip())
                mascara = []
                for _ in range(tamanho):
                    linha = list(map(float, f.readline().strip().split()))
                    mascara.append(linha)
                kernel = np.array(mascara)
        except Exception as e:
            print(f"[ERRO] Erro ao ler o filtro: {e}")
            return

        img_array = np.array(self.imagem)
        resultado = np.zeros_like(img_array)

        for canal in range(3):  # R, G, B
            resultado[:, :, canal] = self.correlacao2d(img_array[:, :, canal], kernel, tipo_filtro)

        self.imagem = Image.fromarray(np.uint8(np.clip(resultado, 0, 255)))

    def correlacao2d(self, canal, kernel, tipo_filtro):
        altura, largura = canal.shape
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        output = np.zeros_like(canal, dtype=float)

        # Ajusta o loop para suportar kernels pares ou Ã­mpares
        for i in range(ph, altura - (kh - ph - 1)):
            for j in range(pw, largura - (kw - pw - 1)):
                regiao = canal[i - ph:i + (kh - ph), j - pw:j + (kw - pw)]
                if regiao.shape == kernel.shape:
                    valor = np.sum(regiao * kernel)
                    output[i, j] = valor
                else:
                    # Caso raro de borda que ainda pode falhar
                    pass

        if "sobel" in tipo_filtro:
            output = np.abs(output)
            output = self.expandir_histograma(output)

        return output

    def expandir_histograma(self, canal):
        min_val = np.min(canal)
        max_val = np.max(canal)
        if max_val - min_val == 0:
            return np.zeros_like(canal)
        canal_norm = (canal - min_val) * 255.0 / (max_val - min_val)
        return canal_norm

# ===========================
# Exemplo de uso
# ===========================
if __name__ == "__main__":
    sistema = SistemaImagem()
    
    # ✅ Use barra normal ou caminho raw (compatível)
    imagem_path = os.path.join("Testes", "testpat.1k.color2.tif")
    filtro_path = os.path.join("Testes", "sobelHorizontal.txt")

    sistema.abrir_imagem(imagem_path)
    sistema.aplicar_filtro_txt(filtro_path)
    sistema.exibir_imagem()
    sistema.salvar_imagem("sobelHorizontal.jpg")
