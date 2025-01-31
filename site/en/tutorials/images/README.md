# 📸 Projeto de Aumento de Dados para Imagens de Flores

Este projeto faz parte das atividades da faculdade e tem como objetivo realizar o **download, extração e aumento de dados** para um conjunto de imagens de flores. O aumento de dados (data augmentation) é uma técnica essencial em visão computacional para ampliar a diversidade do conjunto de treinamento sem coletar novas imagens.

## 🚀 Funcionalidades
- 📥 **Download e extração automática** do dataset de flores do TensorFlow.
- 🔄 **Transformações de aumento de dados**, incluindo:
  - Espelhamento horizontal
  - Rotação aleatória
  - Zoom aleatório
- 🖼️ **Visualização** das imagens antes e depois das transformações.

## 📂 Estrutura do Projeto
```
📦 flower_photos_augmentation
├── 📜 main.py  # Script principal
├── 📂 flower_photos  # Diretório do dataset original
├── 📂 flower_photos_aumentadas  # Diretório das imagens transformadas
├── 📜 README.md  # Documentação do projeto
```

## 📥 Instalação e Uso
1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/flower_photos_augmentation.git
   cd flower_photos_augmentation
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o script**:
   ```bash
   python main.py
   ```

## 🔧 Tecnologias Utilizadas
- 🐍 **Python**
- 📦 **Pillow** (Manipulação de imagens)
- 📦 **Matplotlib** (Visualização de imagens)
- 🌐 **urllib** (Download de arquivos)
- 🗂️ **tarfile** (Extração de arquivos)
- 🏷️ **glob** (Listagem de arquivos)

## 🎯 Objetivo Acadêmico
Este projeto foi desenvolvido para explorar técnicas de **processamento de imagens** e **aprendizado profundo**, garantindo um conjunto de dados mais diversificado para modelos de Machine Learning e Visão Computacional.

📬 **Contato:** Caso tenha dúvidas ou sugestões, sinta-se à vontade para abrir uma issue ou entrar em contato!

---
⭐ Se você gostou deste projeto, não esqueça de dar um **star** no repositório!

