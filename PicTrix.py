import customtkinter as tk
from customtkinter import filedialog
from PIL import Image, ImageTk 
import cv2
import numpy as np
import io

imagem_original = None  # Variável para armazenar a imagem original
imagem_ref = None 
zoom_level = 100

# Função para salvar a imagem modificada
def salvar_imagem():
    global zoom_level

    if imagem_cv2 is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Imagens PNG", "*.png"), ("Todos os arquivos", "*.*")])
        if file_path:
            # Obtém a imagem atual do canvas_imagem
            imagem_canvas = canvas_imagem.postscript(colormode="color")

            # Converte o PS (PostScript) em uma imagem PIL
            imagem_pil = Image.open(io.BytesIO(imagem_canvas.encode("utf-8")))
            # Salva a imagem no formato desejado
            imagem_pil.save(file_path)
            print("Imagem salva com sucesso!")

# Função para carregar uma imagem
def carregar_imagem():
    global imagem_cv2, imagem_original
    file_path = filedialog.askopenfilename()
    if file_path:
        imagem_cv2 = cv2.imread(file_path)
        imagem_original = imagem_cv2.copy()  # Faz uma cópia da imagem original
        mostrar_imagem(imagem_cv2)

# Função para carregar a imagem de referência
def carregar_imagem_referencia():
    global imagem_ref

    file_path = filedialog.askopenfilename()
    if file_path:
        imagem_ref = cv2.imread(file_path)
        #mostrar_imagem(imagem_ref)

# Função para aplicar zoom na imagem
def zoom_imagem():
    global zoom_level
    zoom_input = tk.CTkInputDialog(text="Digite o nível de zoom (em %):", title="Zoom")
    novo_zoom = zoom_input.get_input()
    if novo_zoom is not None:
        try:
            novo_zoom = int(novo_zoom)
            if novo_zoom > 0:
                zoom_level = novo_zoom
                mostrar_imagem(imagem_original)  # Recarregar a imagem original com o novo zoom
            else:
                print("Valor de zoom inválido. O zoom deve ser um número positivo.")
        except ValueError:
            print("Valor de zoom inválido. O zoom deve ser um número inteiro positivo.")

# Função para mostrar a imagem carregada na interface
def mostrar_imagem(imagem):
    global zoom_level
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    altura, largura, _ = imagem.shape
    nova_largura = int(largura * zoom_level / 100)
    nova_altura = int(altura * zoom_level / 100)
    imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))
    imagem_pil = Image.fromarray(imagem_redimensionada)
    imagem_tk = ImageTk.PhotoImage(imagem_pil)
    # Calcula as coordenadas x e y para centralizar a imagem no canvas_imagem
    x = (canvas_imagem.winfo_width() - nova_largura) // 2
    y = (canvas_imagem.winfo_height() - nova_altura) // 2

    # Limpa o canvas 
    canvas_imagem.delete("all")
    # Define o tamanho do canvas com base no tamanho da imagem redimensionada
    canvas_imagem.config(width=nova_largura, height=nova_altura)
    #Cria a imagem centralizada
    canvas_imagem.create_image(x, y, anchor=tk.NW, image=imagem_tk)
    canvas_imagem.image = imagem_tk

# Função para aplicar um filtro de desfoque à imagem
def aplicar_desfoque():
    if imagem_cv2 is not None:
        global imagem_original
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        imagem_desfocada = cv2.GaussianBlur(imagem_cv2, (21, 21), 0)
        mostrar_imagem(imagem_desfocada)

# Função para aplicar filtro da mediana, retirar ruido sal pimenta
def filtroMediana():
    if imagem_cv2 is not None:
        global imagem_original
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        imagem_desfocada = cv2.medianBlur(imagem_cv2, 3)
        mostrar_imagem(imagem_desfocada)

# Função para aplicar um filtro de realce à imagem
def aplicar_realce():
    if imagem_cv2 is not None:
        global imagem_original
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) #Laplaciano
        imagem_realcada = cv2.filter2D(imagem_cv2, -1, kernel)
        mostrar_imagem(imagem_realcada)

# Função para aplicar um filtro de realce sobel
def aplicar_realce_sobel():
    if imagem_cv2 is not None:
        global imagem_original
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        #filtros de sobel para calculo das derivadas parciais
        sobX = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobY = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        Gx = cv2.filter2D(imagem_cv2, cv2.CV_64F, sobX)# gradiente na direção X(linhas)
        Gy = cv2.filter2D(imagem_cv2, cv2.CV_64F, sobY)# gradiente na direção Y(linhas)

        mag = np.sqrt( Gx**2 + Gy**2 ) # magnitude do vetor gradiente

        img_agucada = imagem_cv2 + 0.4 * mag
        img_agucada[img_agucada > 255]=255
        img_agucada = img_agucada.astype(np.uint8)

        mag[mag > 255] = 255
        mag = mag.astype(np.uint8)

        #img_agucada = cv2.cvtColor(img_agucada, cv2.COLOR_BGR2RGB)
        mostrar_imagem(img_agucada)

# Função para aplicar um filtro de realce à imagem
def filtro_gamma():
    if imagem_cv2 is not None:
        global imagem_original
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        gamma = (tk.CTkInputDialog(text="Informe o valor de gamma", title="gamma")).get_input()
        gamma = int(gamma)
        img=imagem_cv2
        c = 255.0 / (255.0**gamma)
        img_gamma = c * (img.astype(np.float64))**gamma
        img_gamma= img_gamma.astype(np.uint8)
        mostrar_imagem(img_gamma)

# Função para aplicar equalização da imagem
def aplicar_equalizacao():
    if imagem_cv2 is not None:
        global imagem_original
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        img=imagem_cv2.copy()
        R = img.shape[0]
        C = img.shape[1]

        #calculo do histograma normalizado (pr)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]) 
        pr = hist/(R*C)

        # cummulative distribution function (CDF)
        cdf = pr.cumsum()
        sk = 255 * cdf
        sk = np.round(sk)

        # criando a imagem de saída
        img_out = np.zeros(img.shape, dtype=np.uint8)
        for i in range(256):
            img_out[img == i] = sk[i]
        mostrar_imagem(img_out)

# Função para aplicar equalização da imagem
def aplicar_equalizacao_por_referencia():
    global imagem_original, imagem_ref
    carregar_imagem_referencia()
    if imagem_cv2 is not None and imagem_ref is not None:
        imagem_original = imagem_cv2.copy()  # Salva a imagem original antes de aplicar o filtro
        img_input=imagem_cv2.copy()#imagem de entrada       
        img_ref =  imagem_ref.copy()

        chans_img = cv2.split(img_input)#separa os canais de cores
        chans_ref = cv2.split(img_ref)

        # iterage nos canais da imagem de entrada e calcula o histograma
        pr = np.zeros((256, 3))
        for chan, n in zip(chans_img, np.arange(3)):
            pr[:,n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()

        # iterage nos canais da imagem de referencia e calcula o histograma
        pz = np.zeros((256, 3))
        for chan, n in zip(chans_ref, np.arange(3)):
            pz[:,n] = cv2.calcHist([chan], [0], None, [256], [0, 256]).ravel()
        
        # calcula as CDFs para a imagem de entrada
        cdf_input = np.zeros((256, 3))
        for i in range(3):
            cdf_input[:,i] = np.cumsum(pr[:,i]) # referencia
        
        # calcula as CDFs para a imagem de referencia
        cdf_ref = np.zeros((256,3))
        for i in range(3):
            cdf_ref[:,i] = np.cumsum(pz[:,i]) # referencia
    

    img_out = np.zeros(img_input.shape) # imagem de saida #shape pega dimensão

    for c in range(3):#corre nos planos de cores
        for i in range(256): #corre na cdf de cada plano da imagem
            diff = np.absolute(cdf_ref[:,c] - cdf_input[i,c])
            indice = diff.argmin()
            img_out[img_input[:,:,c] == i, c] = indice

    img_out = img_out.astype(np.uint8)
    
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    mostrar_imagem(img_out)

# Função para restaurar a imagem original
def restaurar_original():
    global imagem_cv2, imagem_original
    if imagem_original is not None:
        imagem_cv2 = imagem_original.copy()  # Restaura a imagem original
        mostrar_imagem(imagem_cv2)

fatorBrilho = 1.0  # Inicializa o fator de brilho
def ajustar_e_mostrar_brilho():
    global fatorBrilho, imagem_cv2, imagem_original

    brilho_input = tk.CTkInputDialog(text="Digite o nível do fator de multiplicação de brilho (0~2):", title="Brilho")
    novo_fator_brilho = brilho_input.get_input()

    if novo_fator_brilho is not None:
        try:
            novo_fator_brilho = float(novo_fator_brilho)
            if 0 <= novo_fator_brilho <= 2:
                fatorBrilho = novo_fator_brilho
                imagem_nova = ajustar_brilho(imagem_original, fatorBrilho)
                mostrar_imagem(imagem_nova)
            else:
                print("Fator de brilho fora do intervalo válido (0~2).")
        except ValueError:
            print("Valor de fator de brilho inválido. Use um número decimal entre 0 e 2.")

# Função para ajustar o brilho da imagem
def ajustar_brilho(imagem, fator):
    # Verifica se a imagem é uma imagem colorida ou em tons de cinza
    if imagem.ndim == 2:
        # Imagem em tons de cinza
        nova_imagem = imagem.astype(np.float64)
        nova_imagem = nova_imagem * fator
        nova_imagem[nova_imagem > 255] = 255
        nova_imagem[nova_imagem < 0] = 0
        nova_imagem = nova_imagem.astype(np.uint8)
    elif imagem.ndim == 3:
        # Imagem colorida (3 canais: Vermelho, Verde, Azul)
        nova_imagem = imagem.astype(np.float64)
        nova_imagem = nova_imagem * fator
        nova_imagem[nova_imagem > 255] = 255
        nova_imagem[nova_imagem < 0] = 0
        nova_imagem = nova_imagem.astype(np.uint8)
    else:
        raise ValueError("Imagem com número de dimensões não suportado.")

    return nova_imagem

def criar_negativo(imagem):
    if imagem.ndim == 2:
        # Imagem em tons de cinza
        imagem_negativa = 255 - imagem
    elif imagem.ndim == 3:
        # Imagem colorida (3 canais: Vermelho, Verde, Azul)
        imagem_negativa = 255 - imagem
    else:
        raise ValueError("Imagem com número de dimensões não suportado.")

    return imagem_negativa

def aplicar_negativo_e_mostrar():
    global imagem_cv2, imagem_original

    if imagem_cv2 is not None:
        # Salva a imagem original antes de aplicar o efeito
        imagem_original = imagem_cv2.copy()

        # Aplica o efeito de negativo
        imagem_negativa = criar_negativo(imagem_original)

        # Mostra a imagem com o efeito na interface
        mostrar_imagem(imagem_negativa)


# Argumentos Padrões para os botões
btn_args_padrao = {'bg_color': '#190061',
                   'fg_color': '#3500D3'}

def cria_botoes(master, argsPadrao, text, command=None):
    botao = tk.CTkButton(master, **argsPadrao, text=text, command=command)
    return botao

def menu_botoes():
    # Frame para os botões
    frame_botoes = tk.CTkFrame(master=janela, width=160, height=320, fg_color="#190061")
    frame_botoes.pack(side="left", padx=10, pady=10)

    global btn_args_padrao

    # Botão para carregar a imagem
    btn_carregar = cria_botoes(frame_botoes, btn_args_padrao, "Carregar Imagem", command=carregar_imagem)
    btn_carregar.pack(pady=10)

    # Botão para salvar a imagem
    btn_salvar = cria_botoes(frame_botoes, btn_args_padrao, "Salvar Imagem", command=salvar_imagem)
    btn_salvar.pack(pady=10)

    # Botão para restaurar a imagem original
    btn_restaurar = cria_botoes(frame_botoes, btn_args_padrao, "Restaurar Original", command=restaurar_original)
    btn_restaurar.pack(pady=10)

    # Botão para aplicar zoom na imagem
    btn_zoom = cria_botoes(frame_botoes, btn_args_padrao, "Zoom", command=zoom_imagem)
    btn_zoom.pack(padx=10, pady=10)

    # Botão para selecionar filtros
    btn_restaurar = cria_botoes(frame_botoes, btn_args_padrao, "Selecionar Efeitos", command=janela_efeitos)
    btn_restaurar.pack(pady=10)

def janela_efeitos():
    nova_janela = tk.CTkToplevel(janela)
    nova_janela.title("Efeitos")
    global btn_args_padrao
    # Botão para aplicar desfoque
    btn_desfoque = cria_botoes(nova_janela, btn_args_padrao, "Aplicar Desfoque(Gaussian)", command=aplicar_desfoque)
    btn_desfoque.pack(pady=10)

    # Botão para aplicar realce
    btn_realce = cria_botoes(nova_janela, btn_args_padrao, "Aplicar Realce(Laplaciano)", command=aplicar_realce)
    btn_realce.pack(pady=10)

    # Botão para aplicar realce
    btn_realce = cria_botoes(nova_janela, btn_args_padrao, "Aplicar Realce(Sobel)", command=aplicar_realce_sobel)
    btn_realce.pack(pady=10)

    # Botão para aumentar ou diminuir brilho
    btn_zoom = cria_botoes(nova_janela, btn_args_padrao,"Brilho", command=ajustar_e_mostrar_brilho)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Botão para negativo da imagem
    btn_zoom = cria_botoes(nova_janela, btn_args_padrao,"Negativo", command=aplicar_negativo_e_mostrar)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Botão para aplicar filtro da mediana
    btn_zoom = cria_botoes(nova_janela, btn_args_padrao,"Mediana", command=filtroMediana)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Botão para aplicar equalização
    btn_zoom = cria_botoes(nova_janela, btn_args_padrao,"Equalizacao", command=aplicar_equalizacao)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Botão para aplicar especificação
    btn_zoom = cria_botoes(nova_janela, btn_args_padrao,"Especificacao", command=aplicar_equalizacao_por_referencia)
    btn_zoom.pack(side="top", padx=10, pady=10)

    # Botão para aplicar filtro gamma
    btn_zoom = cria_botoes(nova_janela, btn_args_padrao,"Gamma", command=filtro_gamma)
    btn_zoom.pack(side="top", padx=10, pady=10)

if __name__ == "__main__":
    # Configuração da janela principal
    tk.set_default_color_theme("dark-blue")
    janela = tk.CTk()
    janela.title("PicTrix - Editor de Imagens")
    janela.maxsize(width= 1600,height=800)
    janela.minsize(width=600, height=400)
    
    janela.resizable(width=False, height=False)
    menu_botoes()
    # Exibição da imagem carregada no Canvas
    canvas_imagem = tk.CTkCanvas(master=janela, bg="#240090")
    canvas_imagem.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    imagem_cv2 = None

    # Função para redimensionar a imagem quando a janela é redimensionada
    def redimensionar_canvas(event):
        if imagem_original is not None:
            mostrar_imagem(imagem_original)

    # Vincula a função redimensionar_canvas ao evento de redimensionamento da janela
    janela.bind("<Configure>", redimensionar_canvas)

    janela.mainloop()
