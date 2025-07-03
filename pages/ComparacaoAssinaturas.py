import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import cv2
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import gdown
from keras.src.saving import load_model
from st_img_pastebutton import paste
from io import BytesIO
import base64
from keras.src.utils.image_utils import img_to_array
from modelo_similaridade import SignatureVerificationPipeline

@st.cache_data
def load_signature_model(model_path_output, link_drive):
    model_path = model_path_output

    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Baixa o modelo se ainda não estiver salvo localmente
    if not os.path.exists(model_path):
        url = link_drive
        gdown.download(url, model_path, quiet=False, fuzzy=True)

    return load_model(model_path)

class ModeloLimpezaAssinaturas:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.header("Comparador de assinaturas")

        self.model = load_signature_model('modelos_treinados/best_model.h5',
        'https://drive.google.com/file/d/1xCftLBPLD8y89evl4VjL4IyfOU1hkPpx/view?usp=sharing')

        self.input_folder = 'dados/assi_identificadas'
        self.output_folder = 'dados/clean_signatures'
        self.folder_list = [self.input_folder, self.output_folder]

        try:
            os.makedirs('dados/assi_identificadas')
            os.makedirs('dados/clean_signatures')
        except FileExistsError:
            pass

        if 'image_data1' not in st.session_state:
            st.session_state.image_data1 = None
        if 'image_data2' not in st.session_state:
            st.session_state.image_data2 = None
        if 'binary_data1' not in st.session_state:
            st.session_state.binary_data1 = None
        if 'binary_data2' not in st.session_state:
            st.session_state.binary_data2 = None
        if 'image1_pil' not in st.session_state:
            st.session_state.image1_pil = None
        if 'image2_pil' not in st.session_state:
            st.session_state.image2_pil = None

    @staticmethod
    def remove_todos_arquivos_pasta(diretorio):
        for arquivo in os.listdir(diretorio):
            caminho_arquivo = os.path.join(diretorio, arquivo)
            if os.path.isfile(caminho_arquivo):
                os.remove(caminho_arquivo)

    def cria_campo_copy_paste_img(self):
        st.write("Imagem 1:")
        st.session_state.image_data1 = paste(label="Cole uma imagem 1", key="image_clipboard1")
        if st.session_state.image_data1 is not None:
            header1, encoded1 = st.session_state.image_data1.split(",", 1)
            st.session_state.binary_data1 = base64.b64decode(encoded1)
            bytes_data1 = BytesIO(st.session_state.binary_data1)
            st.session_state.image1_pil = Image.open(bytes_data1)
            st.image(bytes_data1, caption="Imagem 1 carregada ✅")
        else:
            st.write("Nenhuma imagem 1 carregada ainda.")

        st.write("Imagem 2:")
        st.session_state.image_data2 = paste(label="Cole uma imagem 2", key="image_clipboard2")
        if st.session_state.image_data2 is not None:
            header2, encoded2 = st.session_state.image_data2.split(",", 1)
            st.session_state.binary_data2 = base64.b64decode(encoded2)
            bytes_data2 = BytesIO(st.session_state.binary_data2)
            st.session_state.image2_pil = Image.open(bytes_data2)
            st.image(bytes_data2, caption="Imagem 2 carregada ✅")
        else:
            st.write("Nenhuma imagem 2 carregada ainda.")

    def salva_imagens(self):
        with open(os.path.join(self.input_folder, 'assinatura_1.png'), 'wb') as f:
            f.write(st.session_state.binary_data1)
        with open(os.path.join(self.input_folder, 'assinatura_2.png'), 'wb') as f:
            f.write(st.session_state.binary_data2)

    @staticmethod
    def resize_with_padding(img, target_size=(128, 128)):
        img = ImageOps.contain(img, target_size, method=Image.Resampling.LANCZOS)
        padded_img = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(255,))
        return padded_img

    @staticmethod
    def verifica_necessidade_limpeza(img_array):
        clf_ruido = load_signature_model("modelos_treinados/best_classificador.h5",
                                         'https://drive.google.com/file/d/1Q5hHhVQwBlxNmqfozpYsnrJUYhMvN8mH/view?usp=sharing')
        pred = clf_ruido.predict(img_array)
        return pred[0] > 0.5

    def save_clean_images(self, input_folder, output_folder, model, limpar_ruidos=True, lista_imgs_ruidosas=None, target_size=(128, 128)):
        if lista_imgs_ruidosas is None:
            lista_imgs_ruidosas = []

        ruidosa = False
        cont = 0

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            original_img = Image.open(os.path.join(input_folder, filename)).convert('L')
            original_size = original_img.size
            padded_img = ModeloLimpezaAssinaturas.resize_with_padding(original_img, target_size)
            img_array = img_to_array(padded_img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            for img in lista_imgs_ruidosas:
                if img == "Imagem 1" and filename == "assinatura_1.png":
                    ruidosa = True
                    break
                elif img == "Imagem 2" and filename == "assinatura_2.png":
                    ruidosa = True
                    break
                else:
                    ruidosa = False

            if limpar_ruidos and ModeloLimpezaAssinaturas.verifica_necessidade_limpeza(img_array)[0]:
                ruidosa = True

            if ruidosa:
                ruidosa=False
                cont += 1
                clean_img = model.predict(img_array)
                clean_img = (clean_img[0] * 255).astype(np.uint8)
                clean_img_resized = cv2.resize(clean_img, original_size, interpolation=cv2.INTER_LINEAR)
                _, _, filtro = ModeloLimpezaAssinaturas.aplica_filtro(self, self.filtro)
                if filtro is not None:
                    try:
                        clean_img_resized = filtro(clean_img_resized)
                    except:
                        try:
                            clean_img_resized = filtro(Image.fromarray(clean_img_resized))
                            clean_img_resized = np.array(clean_img_resized)
                        except:
                            clean_img_resized = filtro(np.array(clean_img_resized))

                cv2.imwrite(os.path.join(output_folder, filename), np.array(clean_img_resized))
            else:

                _, _, filtro = ModeloLimpezaAssinaturas.aplica_filtro(self, self.filtro)

                if filtro is not None:
                    try:
                        padded_img = filtro(padded_img)
                        padded_img = np.array(padded_img)
                    except:
                        padded_img = filtro(np.array(padded_img))

                padded_img = cv2.resize(np.array(padded_img), original_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(output_folder, filename), np.array(padded_img))

        return cont

    @staticmethod
    def plot_assinaturas_antes_depois(input_folder, output_folder):
        c1, c2 = st.columns(2)
        with c1:
            for filename in os.listdir(input_folder):
                original_img = Image.open(os.path.join(input_folder, filename))
                st.image(original_img)
        with c2:
            for filename in os.listdir(output_folder):
                original_img = Image.open(os.path.join(output_folder, filename))
                st.image(original_img)

    @staticmethod
    def filtro_nitidez1(img):
        enhancer = ImageEnhance.Sharpness(img)
        imagem_nitida = enhancer.enhance(2.0)
        return imagem_nitida

    @staticmethod
    def filtro_nitidez2(img):
        enhancer = ImageEnhance.Sharpness(img)
        imagem_nitida = enhancer.enhance(500)
        return imagem_nitida

    @staticmethod
    def filtro_nitidez3(img):
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        blurred = cv2.GaussianBlur(sharpened, (5, 5), sigmaX=0)
        median_filtered = cv2.medianBlur(blurred, ksize=3)
        return median_filtered

    @staticmethod
    def executa_modelo_comparacao_assinaturas(img1_path, img2_path, progress_bar):
        pipeline = SignatureVerificationPipeline()
        pipeline.load_pipeline("modelos_treinados/SVM_assinaturas.pkl")
        progress_bar.progress(50, text="Verificando autenticidade da assinatura...")
        prediction, prob = pipeline.predict(img1_path, img2_path)
        progress_bar.progress(100, text="Verificando autenticidade da assinatura...")
        st.write(f"🔍 Probabilidade de ser forjada: {prob:.4f}")
        st.write(f"✅ Resultado: {'Genuína ✅' if prediction == 0 else 'Forjada ❌'}")
        if prediction == 0:
            st.balloons()

    def execucao_personalizada(self):
        st.write("Selecione se deseja executar de forma personalizada ou automática")
        self.option_exec = st.selectbox("Tipo de execução:", options=["Automática", "Personalizada"], index=0)

    def aplica_filtro(self, filtro):
        if filtro == "Filtro 1":
            imagem_nitida = ModeloLimpezaAssinaturas.filtro_nitidez1(st.session_state.image1_pil)
            imagem_nitida2 = ModeloLimpezaAssinaturas.filtro_nitidez1(st.session_state.image2_pil)
            filtro_aplicar = self.filtro_nitidez1
        elif filtro == "Filtro 2":
            imagem_nitida = ModeloLimpezaAssinaturas.filtro_nitidez2(st.session_state.image1_pil)
            imagem_nitida2 = ModeloLimpezaAssinaturas.filtro_nitidez2(st.session_state.image2_pil)
            filtro_aplicar = self.filtro_nitidez2
        elif filtro == "Filtro 3":
            imagem_nitida = ModeloLimpezaAssinaturas.filtro_nitidez3(np.array(st.session_state.image1_pil))
            imagem_nitida2 = ModeloLimpezaAssinaturas.filtro_nitidez3(np.array(st.session_state.image2_pil))
            filtro_aplicar = self.filtro_nitidez3
        else:
            imagem_nitida = st.session_state.image1_pil
            imagem_nitida2 = st.session_state.image2_pil
            filtro_aplicar = None

        return imagem_nitida, imagem_nitida2, filtro_aplicar

    def cria_campo_filtros_nitidez(self):
        self.filtro = st.radio("Escolha um filtro de nitidez:", ("Nenhum", "Filtro 1", "Filtro 2", "Filtro 3"))
        imagem_nitida, imagem_nitida2, self.filtro = ModeloLimpezaAssinaturas.aplica_filtro(self, self.filtro)
        img1_div, img2_div = st.columns(2)

        with img1_div:
            st.image(imagem_nitida, caption="Imagem com Filtro Aplicado", use_container_width=True)
        with img2_div:
            st.image(imagem_nitida2, caption="Imagem com Filtro Aplicado", use_container_width=True)

    def cria_fluxo_execucao_personalizado(self):  # noqa: E305
        if st.session_state.image_data1 and st.session_state.image_data2:
            ModeloLimpezaAssinaturas.salva_imagens(self)

            _, c2, _ = st.columns([0.2, 0.6, 0.2])
            with c2:
                imgs_ruidosas_selecionadas = st.multiselect(
                    'Quais imagens são ruidosas?', options=['Imagem 1', 'Imagem 2']
                )

            ModeloLimpezaAssinaturas.cria_campo_filtros_nitidez(self)

            _, div_btn, _ = st.columns(3)
            with div_btn:
                self.btn_exec_person = st.button('Executar', type='primary')

            if self.btn_exec_person:
                with st.spinner('Processando...'):
                    _ = ModeloLimpezaAssinaturas.save_clean_images(
                        self, self.input_folder, self.output_folder, self.model,
                        lista_imgs_ruidosas=imgs_ruidosas_selecionadas, limpar_ruidos=False
                    )

                if len(imgs_ruidosas_selecionadas) > 0:
                    with st.expander('Realizando limpeza das assinaturas...'):
                        ModeloLimpezaAssinaturas.plot_assinaturas_antes_depois(
                            self.input_folder, self.output_folder
                        )

                progress_bar = st.progress(0, text='Verificando autenticidade da assinatura...')
                ModeloLimpezaAssinaturas.executa_modelo_comparacao_assinaturas(
                    os.path.join(self.output_folder, 'assinatura_1.png'),
                    os.path.join(self.output_folder, 'assinatura_2.png'),
                    progress_bar
                )

                for folder in self.folder_list:
                    ModeloLimpezaAssinaturas.remove_todos_arquivos_pasta(folder)

    def cria_fluxo_execucao_automatico(self):
        if st.session_state.image_data1 and st.session_state.image_data2:
            ModeloLimpezaAssinaturas.salva_imagens(self)

            if st.button("Executar", type="primary"):
                with st.spinner("Processando..."):
                    self.filtro = "Filtro 1"
                    qtd_sig_ruidosas_identificadas = ModeloLimpezaAssinaturas.save_clean_images(
                        self, self.input_folder, self.output_folder, self.model,
                        #lista_imgs_ruidosas=["Imagem 1", "Imagem 2"],
                        limpar_ruidos=True)

                st.write(f"✅ Foram identificadas {qtd_sig_ruidosas_identificadas} assinatura(s) ruidosa(s)")

                if qtd_sig_ruidosas_identificadas > 0:
                    with st.expander("🧼 Realizando limpeza das assinaturas"):
                        ModeloLimpezaAssinaturas.plot_assinaturas_antes_depois(self.input_folder,
                                                                               self.output_folder)

                progress_bar = st.progress(0, text="Verificando autenticidade da assinatura...")
                ModeloLimpezaAssinaturas.executa_modelo_comparacao_assinaturas(
                    os.path.join(self.output_folder, "assinatura_1.png"),
                    os.path.join(self.output_folder, "assinatura_2.png"),
                    progress_bar
                )

        for folder in self.folder_list:
            ModeloLimpezaAssinaturas.remove_todos_arquivos_pasta(folder)

    def orquestradora(self):
        ModeloLimpezaAssinaturas.execucao_personalizada(self)
        ModeloLimpezaAssinaturas.cria_campo_copy_paste_img(self)

        if self.option_exec == "Personalizada":
            ModeloLimpezaAssinaturas.cria_fluxo_execucao_personalizado(self)
        else:
            ModeloLimpezaAssinaturas.cria_fluxo_execucao_automatico(self)

if __name__ == "__main__":
    model = ModeloLimpezaAssinaturas()
    model.orquestradora()