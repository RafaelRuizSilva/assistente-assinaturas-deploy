
# 🖋️ Assistente de Assinaturas

Um sistema inteligente construído com Python e Streamlit para **análise e verificação de autenticidade de assinaturas**.  
O projeto combina modelos de deep learning com uma interface visual intuitiva para detectar ruídos, limpar assinaturas e validar sua autenticidade.

🔗 Acesse o app: [Signature Assistant](https://signature-assistant-dep.streamlit.app/)

---

## 🚀 Funcionalidades

O assistente realiza **3 etapas principais** com modelos especializados:

1. 🔍 **Detecção de Ruído em Assinaturas**  
   Modelo **CNN** que identifica se uma assinatura está ruidosa (borrada, com fundo poluído ou artefatos visuais).

2. 🧹 **Limpeza de Ruído (Denoising)**  
   Modelo **ResUNet** para reconstrução de imagens limpas a partir de versões ruidosas.  
   Arquitetura baseada em U-Net com blocos residuais:  
   <img width="574" height="1048" alt="image" src="https://github.com/user-attachments/assets/7d676a98-6eb7-4452-8cf4-56311de87439" />


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
<img width="733" height="850" alt="image" src="https://github.com/user-attachments/assets/27bb90fa-303f-49cf-aace-1cba4bc39c5d" />


#### Resultado final com probabilidade de falsificação:
<img width="727" height="597" alt="image" src="https://github.com/user-attachments/assets/8a458827-eb31-4568-b9e4-03548e836317" />


---

## 📁 Estrutura dos Modelos

- Os arquivos `.h5` dos modelos são carregados automaticamente via Google Drive, sem necessidade de subir localmente
- A estrutura ResUNet usada para denoising está descrita na imagem da arquitetura

---

## 📄 Licença

Este projeto é de uso acadêmico e educacional. Sinta-se livre para usar como base para projetos relacionados à biometria, forense ou verificação de identidade.
