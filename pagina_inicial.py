import warnings
import logging
import streamlit as st

def configs_iniciais():  # usage  rsrhwvv
    warnings.filterwarnings('ignore')

    logging.basicConfig(filename='errors.log', level=logging.ERROR)

    st.set_page_config(
        page_title='Limpador de assinaturas',
        page_icon='🖋️',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    st.title("🖋️ Comparador Inteligente de Assinaturas")

    st.write("""
    Bem-vindo ao **Assistente de Assinaturas**, uma ferramenta inteligente desenvolvida com técnicas avançadas de Visão Computacional e Aprendizado de Máquina.

    Este sistema foi projetado para **auxiliar na análise forense de assinaturas**, oferecendo uma abordagem automatizada em 3 etapas:
    """)

    st.markdown("""
    1. **Detecção de Ruído**  
       Identificamos se as assinaturas enviadas estão com ruídos visuais (como linhas, fundos e artefatos).

    2. **Limpeza da Imagem (Denoising)**  
       Utilizamos uma rede neural profunda (**ResUNet**) para limpar a assinatura e reconstruir sua versão idealizada.

    3. **Verificação de Autenticidade**  
       Comparamos as duas assinaturas para estimar se são **genuínas ou forjadas**, com base em modelos treinados com **ResNet50 e SVM**.
    """)

    st.write("---")

    st.subheader("📌 Como usar")

    st.markdown("""
    - Cole ou envie **duas imagens de assinatura**
    - Escolha o tipo de execução: **Automática** ou **Personalizada**
    - Visualize o processo de análise e o resultado final
    """)

    st.write("---")

    st.caption(
        "⚠️ Este projeto tem fins educacionais e de demonstração de técnicas modernas de Deep Learning aplicadas à verificação de assinaturas.")

    #add_custom_css()

if __name__ == '__main__':
    configs_iniciais()
