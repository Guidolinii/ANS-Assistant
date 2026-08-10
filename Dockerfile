# Estágio de construção e execução do ANS-Assistant
# Imagem base Python 3.12 (futura implementação)
# FROM python:3.12-slim

# Definir diretório de trabalho
# WORKDIR /app

# Copiar arquivos de dependências
# COPY requirements.txt .

# Instalar dependências
# RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
# COPY . .

# Expor a porta da aplicação (ex.: 8501 para Streamlit)
# EXPOSE 8501

# Comando de inicialização
# CMD ["streamlit", "run", "app/ui/streamlit_app.py"]
