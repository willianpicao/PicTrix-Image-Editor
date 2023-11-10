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

## ---------

# PicTrix - Image Editor

**PicTrix** is a basic image editing tool that explores image processing concepts. The application is developed in Python and uses various libraries for image manipulation.

## Tools Used

- **customtkinter:** A customized version of the Tkinter library for creating graphical interfaces.
- **PIL (Python Imaging Library):** Used for opening, manipulating, and saving various image file formats.
- **OpenCV (Open Source Computer Vision Library):** Employed for image processing tasks such as filters and transformations.
- **NumPy:** Used for numerical operations and array manipulation in the context of image processing.

## Overview of Functionality

### Image Loading and Saving
- **Load Image:** Allows users to open an image file from their local system.
- **Save Image:** Enables users to save the edited image with specified parameters.

### Image Display and Manipulation
- **Zoom:** Allows users to zoom in or out on the image.
- **Restore Original:** Reverts the image to its original state.
- **Brightness Adjustment:** Adjusts the brightness of the image.

### Filters and Effects
- **Blur (Gaussian):** Applies a Gaussian blur filter to the image.
- **Enhancement (Laplacian):** Enhances the image using a Laplacian filter.
- **Enhancement (Sobel):** Applies Sobel enhancement to the image.
- **Negative:** Creates a negative version of the image.
- **Median Filter:** Removes salt and pepper noise using a median filter.
- **Equalization:** Enhances the contrast of the image through histogram equalization.
- **Specification:** Applies equalization based on a reference image.
- **Gamma Filter:** Adjusts the gamma of the image based on user input.
- **Canny Edge Detection:** Applies Canny edge detection to the image.
- **Hough Transform:** Uses the Hough Transform to detect lines in the image.

## Usage Instructions

1. **Load Image:** Click "Load Image" to select an image from your local system.
2. **Apply Filters:** Use the "Select Effects" button to access a menu with various filter options.
3. **Save Image:** Click "Save Image" to save the edited image with specific parameters.
4. **Zoom:** Zoom in or out on the image using the "Zoom" button.
5. **Restore Original:** Revert the image to its original state with "Restore Original."

Feel free to explore and experiment with different filters and settings to enhance your images!

*Note: Make sure to have the necessary libraries (customtkinter, Pillow, OpenCV, NumPy) installed before running the application.*

