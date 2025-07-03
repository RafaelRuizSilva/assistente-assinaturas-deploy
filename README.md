
# 🖋️ Assistente de Assinaturas

Um sistema inteligente construído com Python e Streamlit para **análise e verificação de autenticidade de assinaturas**.  
O projeto combina modelos de deep learning com uma interface visual intuitiva para detectar ruídos, limpar assinaturas e validar sua autenticidade.

![Interface do Assistente](./assets/interface.png)

---

## 🚀 Funcionalidades

O assistente realiza **3 etapas principais** com modelos especializados:

1. 🔍 **Detecção de Ruído em Assinaturas**  
   Modelo **CNN** que identifica se uma assinatura está ruidosa (borrada, com fundo poluído ou artefatos visuais).

2. 🧹 **Limpeza de Ruído (Denoising)**  
   Modelo **ResUNet** para reconstrução de imagens limpas a partir de versões ruidosas.  
   Arquitetura baseada em U-Net com blocos residuais:  
   ![Arquitetura ResUNet](./assets/resunet.png)

3. ✅ **Verificação de Autenticidade**  
   Arquitetura **ResNet50 + SVM (RBF kernel)** treinada para identificar se duas assinaturas são **genuínas** ou **forjadas**.

---

## 🛠 Tecnologias Utilizadas

- Python 3.11
- Streamlit (interface)
- TensorFlow / Keras (modelos CNN, ResUNet, ResNet50)
- OpenCV (pré-processamento de imagem)
- scikit-learn (SVM)
- PIL, NumPy, etc.

---

## 📦 Instalação

Clone o repositório:

```bash
git clone https://github.com/RafaelRuizSilva/assistente-assinaturas-deploy.git
cd assistente-assinaturas-deploy
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

> ⚠️ Certifique-se de que está usando **Python 3.11**.

---

## ▶️ Execução

Para rodar o assistente localmente, use:

```bash
streamlit run pagina_inicial.py
```

O app abrirá no navegador e permitirá que você cole ou selecione imagens de assinaturas para análise.

---

## 📷 Exemplos

#### Upload e análise de assinaturas:
![Etapas da análise](./assets/analise.png)

#### Resultado final com probabilidade de falsificação:
![Resultado](./assets/resultado.png)

---

## 📁 Estrutura dos Modelos

- Os arquivos `.h5` dos modelos são carregados automaticamente via Google Drive, sem necessidade de subir localmente
- A estrutura ResUNet usada para denoising está descrita na imagem da arquitetura

---

## 📄 Licença

Este projeto é de uso acadêmico e educacional. Sinta-se livre para usar como base para projetos relacionados à biometria, forense ou verificação de identidade.
