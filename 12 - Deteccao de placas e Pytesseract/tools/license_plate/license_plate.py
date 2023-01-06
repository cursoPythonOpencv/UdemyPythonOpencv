# Imports
import cv2
import numpy as np
import imutils
from collections import namedtuple
from skimage import measure
from skimage import segmentation
from skimage.filters import threshold_local
from imutils import perspective

# Define um objeto namedtuple para armazenar a placa
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

class LicensePlateDetector:
    def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40):
        # Armazena a imagem para detectar placas, largura e altura mínima da região da placa de licença, 
        # número de caracteres a serem detectados na placa e a largura mínima dos caracteres extraídos
        self.image = image
        self.minPlateW = minPlateW
        self.minPlateH = minPlateH
        self.numChars = numChars
        self.minCharW = minCharW

    def detect_plates(self):
        # Detecta regiões da placa na imagem com op. morfologicas
        lpRegions = self.morph_operations()

        # Loop sobre as regiões da placa
        for lpRegion in lpRegions:
            # Detecta caracteres candidatos na região atual da placa 
            lp = self.detect_regions_candidate(lpRegion)

            # Somente continua se os caracteres foram detectados com sucesso
            if lp.success:
                # Recorta os candidatos em caracteres
                chars = self.cut_roi(lp)

                yield (lpRegion, chars)

    def morph_operations(self):
        # Inicializa os kernels retangulares e quadrados a serem aplicados à imagem e inicializa a lista das regiões da placa
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        regions = []

        # Converte a imagem em escala de cinza e aplica a operação blackhat
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        # Encontra regiões na imagem que sejam leves
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
        light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]

        # Calcula a representação de gradiente de Scharr da imagem do blackhat na direção x e dimensiona a imagem resultante na faixa [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        # Blur da representação do gradiente, aplica uma operação de fechamento e threshold a imagem usando o método de Otsu
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Realiza uma série de erosões e dilatações na imagem
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Obtém o bitwise 'e' entre as regiões 'light' da imagem, então executa outra série de erosões e dilatação
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=1)

        # Encontre contornos na imagem de limite
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] 

        # Loop sobre os contornos
        for c in cnts:
            # Obtém a caixa delimitadora associada ao contorno e calcula a área e a relação de aspecto
            (w, h) = cv2.boundingRect(c)[2:]
            aspectRatio = w / float(h)

            # Calcula a caixa de delimitação da região
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.cv.BoxPoints(rect)) if imutils.is_cv2() else cv2.boxPoints(rect)

            # Assegura de que a relação de aspecto, largura e altura da caixa delimitadora se enquadrem em limites toleráveis e, 
            # em seguida, atualiza a lista das regiões da placa
            if (aspectRatio > 3 and aspectRatio < 6) and h > self.minPlateH and w > self.minPlateW:
                regions.append(box)

        # Retorna a lista das regiões da placa
        return regions

    def detect_regions_candidate(self, region):
        # Aplica uma transformação de 4 pontos para extrair a placa
        plate = perspective.four_point_transform(self.image, region)
        cv2.imshow("1 - Plate Region with Perspective Transform", imutils.resize(plate, width=400))

        # Extrai o componente Value do espaço de cores HSV e aplica limiar local para revelar os caracteres na placa
        hsv_value = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
        tresh_hsv = threshold_local(hsv_value, 17, offset=15, method="gaussian")
        thresh = (hsv_value > tresh_hsv).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)

        # Redimensiona a região da placa para um tamanho canônico
        plate = imutils.resize(plate, width=400)
        thresh = imutils.resize(thresh, width=400)
        cv2.imshow("2 - License Plate Threshold - Resized", thresh)

        # Realiza uma análise de componentes conectados e inicializa a máscara para armazenar 
        # os locais dos caracteres candidatos 
        labels = measure.label(thresh)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")

        # Loop sobre os componentes únicos
        for label in np.unique(labels):
            # Se este é o rótulo de fundo, ignore-o
            if label == 0:
                continue

            # Caso contrário, construa a máscara de etiqueta para exibir apenas os componentes conectados 
            # para o rótulo atual e, em seguida, encontre contornos na máscara de etiqueta
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] 

            # Certifique-se de que pelo menos um contorno foi encontrado na máscara
            if len(cnts) > 0:
                # Obter o maior contorno que corresponde ao componente na máscara, depois obter 
                # a caixa delimitadora para o contorno
                c = max(cnts, key=cv2.contourArea)
                (_, _, boxW, boxH) = cv2.boundingRect(c)

                # Calcular a relação de aspecto, solidez e altura para o componente
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])

                # Determine se a relação de aspecto, solidez e altura do contorno passam os testes de regras
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                # Verifique se o componente passa todos os testes
                if keepAspectRatio and keepSolidity and keepHeight:
                    # Calcula o casco convexo do contorno e desenha na máscara dos caracteres candidatos 
                    hull = cv2.convexHull(c)
                    cv2.drawContours(charCandidates, [hull], -1, 255, -1)

        # Limpa os pixels que tocam as bordas dos caracteres candidatos mascaram e detectam contornos na máscara de candidatos
        charCandidates = segmentation.clear_border(charCandidates)
        cnts = cv2.findContours(charCandidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        cv2.imshow("3 - Original Candidates", charCandidates)

        # Se houver mais caracteres candidatos do que o número fornecido, então poda os candidatos
        if len(cnts) > self.numChars:
            (charCandidates, cnts) = self.pruneCandidates(charCandidates, cnts)
            cv2.imshow("4 - Pruned Candidates", charCandidates)

        # Obtém o bitwise AND da imagem de limite e imagem de limite em bruto para obter uma segmentação mais limpa dos caracteres
        thresh = cv2.bitwise_and(thresh, thresh, mask=charCandidates)
        cv2.imshow("5 - Char Threshold", thresh)

        # Retorna o objeto da região da placa que contém a placa, a placa limiar e os caracteres candidatos 
        return LicensePlate(success=True, plate=plate, thresh=thresh, candidates=charCandidates)

    def pruneCandidates(self, charCandidates, cnts):
        # Inicializa a máscara de candidatos podados e a lista de dimensões
        prunedCandidates = np.zeros(charCandidates.shape, dtype="uint8")
        dims = []

        # Loop sobre os contornos
        for c in cnts:
            # Calcula a caixa delimitadora para o contorno e atualiza a lista de dimensões
            (_, boxY, _, boxH) = cv2.boundingRect(c)
            dims.append(boxY + boxH)

        # Converte as dimensões em uma matriz NumPy e inicializa a lista de diferenças e contornos selecionados
        dims = np.array(dims)
        diffs = []
        selected = []

        # Loop sobre as dimensões
        for i in range(0, len(dims)):
            # Calcula a soma das diferenças entre a dimensão atual e todas as outras dimensões e, em seguida, atualiza a lista de diferenças
            diffs.append(np.absolute(dims - dims[i]).sum())

        # Encontra o número superior de candidatos com as dimensões mais semelhantes e controla os contornos selecionados
        for i in np.argsort(diffs)[:self.numChars]:
            # Desenha o contorno na máscara de candidatos podados e adiciona à lista de contornos selecionados
            cv2.drawContours(prunedCandidates, [cnts[i]], -1, 255, -1)
            selected.append(cnts[i])

        # Retorna uma tupla da máscara de candidatos podados e contornos selecionados
        return (prunedCandidates, selected)

    def cut_roi(self, lp):
        # Detecta contornos nos candidatos e inicializa a lista de caixas de delimitação e lista de caracteres extraídos
        cnts = cv2.findContours(lp.candidates.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        boxes = []
        chars = []

        # Loop sobre os contornos
        for c in cnts:
            # Calcula a caixa delimitadora para o contorno, mantendo a largura mínima
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            dX = min(self.minCharW, self.minCharW - boxW) // 2
            boxX -= dX
            boxW += (dX * 2)

            # Atualiza a lista de caixas de delimitação
            boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

        # Classifica as caixas delimitadoras da esquerda para a direita
        boxes = sorted(boxes, key=lambda b:b[0])

        # Loop sobre as caixas de delimitação iniciadas
        for (startX, startY, endX, endY) in boxes:
            # Extrai o ROI da placa e atualiza a lista de caracteres
            chars.append(lp.thresh[startY:endY, startX:endX])

        # Retorna a lista de caracteres
        return chars

    @staticmethod
    def preprocessChar(char):
        # Encontra o maior contorno no caracter, obtém sua caixa delimitadora e corta
        cnts = cv2.findContours(char.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0]
        if len(cnts) == 0:
            return None
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        char = char[y:y + h, x:x + w]

        # Retorna o caracter pré-processado
        return char