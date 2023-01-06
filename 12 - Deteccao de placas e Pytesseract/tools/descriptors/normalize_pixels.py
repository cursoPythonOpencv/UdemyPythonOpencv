# Imports
import numpy as np
import cv2

class NormalizePixels:
	"""Objetivo dessa classe é extrair as features da imagem capturada.
	As features são os pixels nao-zero normalizados pelo total de pixels"""
	def __init__(self, size_blocks=((5, 5),)):
		self.size_target = (30, 15)
		self.size_blocks = size_blocks

	def describe(self, image):
		image = cv2.resize(image, (self.size_target[1], self.size_target[0]))
		features = []

		# Loop sobre os tamanhos dos blocos
		for (blockW, blockH) in self.size_blocks:
			# Loop sobre a imagem para o tamanho atual do bloco
			for y in range(0, image.shape[0], blockH):
				for x in range(0, image.shape[1], blockW):
					roi = image[y:y + blockH, x:x + blockW]  # Roi da imagem atual
					total = cv2.countNonZero(roi) / float(roi.shape[0] * roi.shape[1]) # Num. total de pixels não-zero. Normaliza pelo total do bloco
					features.append(total)  # Vetor de caracteristicas

		# Retorna os recursos
		return np.array(features)