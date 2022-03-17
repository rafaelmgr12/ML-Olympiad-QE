#!/bin/bash

echo "Inicio do treinamento dos modelos."
echo "=================================="
echo ""
echo "Treinando modelo Cie. Natureza"
python3 train_predict_natureza.py
echo "Treinando modelo Cie. Humanas"
python3 train_predict_humanas.py
echo "Treinando modelo Linguagens"
python3 train_predict_linguagens.py
echo "Treinando modelo Matematica"
python3 train_predict_matematica.py
echo "Treinando modelo Redacao"
python3 train_predict_redacao.py
#python3 train_predict_redacao_rfclf.py

echo ""
echo "Treino dos modelos concluido com sucesso."
echo "========================================="
