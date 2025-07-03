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

    st.header(f' Bem Vindo ao Extrator de Assinaturas')

    st.write("""
    Esse projeto foi desenvolvido para auxiliar na identificação de assinatura(s) de uma imagem
    e no tratamento dos ruídos nela(s) presente(s).
    """, unsafe_allow_html=True)

    #add_custom_css()

if __name__ == '__main__':
    configs_iniciais()
