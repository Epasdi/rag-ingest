FROM python:3.10-slim

# Instalar dependencias del sistema (necesarias para compilar librerías)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar y instalar dependencias primero (para caché eficiente)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puerto interno (solo para comunicación interna en Coolify)
EXPOSE 8765

# Comando para ejecutar la API en el puerto 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]