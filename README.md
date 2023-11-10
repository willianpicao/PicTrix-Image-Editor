# PicTrix - Editor de Imagens

O **PicTrix** é uma ferramenta básica de edição de imagens que explora conceitos de processamento de imagens. O aplicativo é desenvolvido em Python e utiliza várias bibliotecas para manipulação de imagens.

## Ferramentas Utilizadas

- **customtkinter:** Uma versão personalizada da biblioteca Tkinter para criar interfaces gráficas.
- **PIL (Python Imaging Library):** Usada para abrir, manipular e salvar diversos formatos de arquivo de imagem.
- **OpenCV (Open Source Computer Vision Library):** Empregada para tarefas de processamento de imagem, como filtros e transformações.
- **NumPy:** Utilizado para operações numéricas e manipulação de arrays no contexto do processamento de imagens.

## Visão Geral da Funcionalidade

### Carregamento e Salvamento de Imagens
- **Carregar Imagem:** Permite que os usuários abram um arquivo de imagem de seu sistema local.
- **Salvar Imagem:** Possibilita que os usuários salvem a imagem editada com parâmetros especificados.

### Exibição e Manipulação de Imagens
- **Zoom:** Permite que os usuários aumentem ou diminuam o zoom na imagem.
- **Restaurar Original:** Reverte a imagem para seu estado original.
- **Ajuste de Brilho:** Ajusta o brilho da imagem.

### Filtros e Efeitos
- **Desfoque (Gaussiano):** Aplica um filtro de desfoque gaussiano à imagem.
- **Aprimoramento (Laplaciano):** Aprimora a imagem usando um filtro laplaciano.
- **Aprimoramento (Sobel):** Aplica aprimoramento de Sobel à imagem.
- **Negativo:** Cria uma versão negativa da imagem.
- **Filtro Mediano:** Remove ruídos tipo sal e pimenta usando um filtro mediano.
- **Equalização:** Aprimora o contraste da imagem por meio da equalização do histograma.
- **Especificação:** Aplica equalização com base em uma imagem de referência.
- **Filtro Gama:** Ajusta o gama da imagem com base na entrada do usuário.
- **Detecção de Bordas Canny:** Aplica a detecção de bordas Canny à imagem.
- **Transformada de Hough:** Utiliza a Transformada de Hough para detectar linhas na imagem.

## Instruções de Uso

1. **Carregar Imagem:** Clique em "Carregar Imagem" para selecionar uma imagem em seu sistema local.
2. **Aplicar Filtros:** Use o botão "Selecionar Efeitos" para acessar um menu com várias opções de filtro.
3. **Salvar Imagem:** Clique em "Salvar Imagem" para salvar a imagem editada com parâmetros específicos.
4. **Zoom:** Aumente ou diminua o zoom na imagem usando o botão "Zoom".
5. **Restaurar Original:** Reverta a imagem para seu estado original com "Restaurar Original".

Sinta-se à vontade para explorar e experimentar com diferentes filtros e configurações para aprimorar suas imagens!

*Observação: Certifique-se de ter as bibliotecas necessárias (customtkinter, Pillow, OpenCV, NumPy) instaladas antes de executar o aplicativo.*
