import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Variables globales
modelo_entrenado = False
modelo = None
loss = []
prY = 0
num_layers = 0
neuronas_por_capa = []
rangos_fechas = []

# Variables globales para MNIST
modelo_mnist = None
mnist_entrenado = False

# Función para entrenar el modelo
def entrenar_modelo(data):
    global modelo, loss, modelo_entrenado
    
    Entrada = data.iloc[:, 0].values
    Salida = data.iloc[:, 1].values
    
    # Normalizando datos de entrada
    trX = (Entrada - np.min(Entrada)) / (np.max(Entrada) - np.min(Entrada))
    trY = (Salida - np.min(Salida)) / (np.max(Salida) - np.min(Salida))

    # Guardamos los valores de normalización en la sesión
    st.session_state["x_min"] = np.min(Entrada)
    st.session_state["x_max"] = np.max(Entrada)
    st.session_state["y_min"] = np.min(Salida)
    st.session_state["y_max"] = np.max(Salida)

    
    # Construcción del modelo MLP
    modelo = keras.Sequential()
    modelo.add(layers.Dense(neuronas_por_capa[0], activation='relu', input_shape=(1,)))
    for n in neuronas_por_capa[1:]:
        modelo.add(layers.Dense(n, activation='relu'))
    modelo.add(layers.Dense(1))  # Capa de salida

    # Compilación del modelo
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.text("Entrenando modelo...")

    # Entrenamiento del modelo
    epochs = 100  # Puedes ajustar este valor
    history = modelo.fit(trX, trY, epochs=epochs, verbose=0, callbacks=[
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch + 1) / epochs)
        )
    ])
    
    # Guardar la pérdida
    loss = history.history['loss']

    # Finalizar pantalla de carga
    progress_bar.empty()
    status_text.text("Entrenamiento finalizado.")

    # Guardar modelo entrenado en sesión
    st.session_state["modelo"] = modelo
    st.session_state["entrenado"] = True

    # Graficar resultados
    Y_pred = modelo.predict(trX)

    fig, ax = plt.subplots()
    ax.scatter(trX, trY, label="Datos reales")
    ax.scatter(trX, Y_pred, color='r', label="Predicciones")
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Regresión con Perceptrón Multicapa")
    st.pyplot(fig)

# Función para cargar y preprocesar datos MNIST
def cargar_mnist(archivo):
    try:
        # Asumimos que el archivo es un CSV con valores separados por comas
        # donde la primera columna es la etiqueta y las restantes son los píxeles
        data = pd.read_csv(archivo, header=None)
        
        # Separar etiquetas y características
        y = data.iloc[:, 0].values  # Primera columna son las etiquetas
        X = data.iloc[:, 1:].values / 255.0  # Resto son píxeles, normalizados a [0,1]
        
        # Convertir etiquetas a formato one-hot
        y_one_hot = keras.utils.to_categorical(y, 10)
        
        # Guardar en session_state para uso posterior
        st.session_state["mnist_X"] = X
        st.session_state["mnist_y"] = y
        st.session_state["mnist_y_one_hot"] = y_one_hot
        
        return X, y, y_one_hot
    except Exception as e:
        st.error(f"Error al cargar el archivo MNIST: {str(e)}")
        return None, None, None

# Función para entrenar modelo MNIST
def entrenar_modelo_mnist(X, y_one_hot, num_layers, neuronas_por_capa):
    global modelo_mnist, mnist_entrenado
    
    # Crear modelo
    modelo_mnist = keras.Sequential()
    
    # Capa de entrada (flatten)
    modelo_mnist.add(layers.Flatten(input_shape=(X.shape[1],)))
    
    # Capas ocultas
    for i in range(num_layers):
        modelo_mnist.add(layers.Dense(neuronas_por_capa[i], activation='relu'))
        
    # Capa de salida
    modelo_mnist.add(layers.Dense(10, activation='softmax'))
    
    # Compilar modelo
    modelo_mnist.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.text("Entrenando modelo MNIST...")
    
    # Entrenamiento
    epochs = 10  # Ajustable
    batch_size = 32  # Ajustable
    
    history = modelo_mnist.fit(
        X, y_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch + 1) / epochs)
            )
        ]
    )
    
    # Finalizar pantalla de carga
    progress_bar.empty()
    status_text.text("Entrenamiento MNIST finalizado.")
    
    # Guardar modelo en session_state
    st.session_state["modelo_mnist"] = modelo_mnist
    st.session_state["mnist_entrenado"] = True
    mnist_entrenado = True
    
    # Mostrar precisión
    val_acc = history.history['val_accuracy'][-1]
    st.success(f"Precisión en validación: {val_acc*100:.2f}%")
    
    return modelo_mnist, history


def generar_formato_datos():
    data = {
        "Fecha": pd.date_range(start="2025-01-01", periods=10, freq='D').strftime('%Y%m%d'),
        "TRM": np.random.uniform(4000, 4500, 10)
    }
    df = pd.DataFrame(data)
    return df

def descargar_formato():
    df = generar_formato_datos()
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    col1, col2 = st.columns(2)

    col1.download_button(label="Descargar formato en Excel", data=excel_buffer, file_name="formato_trm.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    col2.download_button(label="Descargar formato en CSV", data=csv_buffer, file_name="formato_trm.csv", mime="text/csv")
    
def predecir_rango(fecha_inicio, fecha_fin):
    if "entrenado" in st.session_state and st.session_state["entrenado"]:
        try:
            modelo = st.session_state["modelo"]
            x_min, x_max = st.session_state["x_min"], st.session_state["x_max"]
            y_min, y_max = st.session_state["y_min"], st.session_state["y_max"]
            
            fechas = np.arange(fecha_inicio, fecha_fin + 1, 1)  # Generar rango de fechas
            fechas_norm = (fechas - x_min) / (x_max - x_min)  # Normalizar
            fechas_tensor = tf.convert_to_tensor(fechas_norm.reshape(-1, 1), dtype=tf.float32)
            
            predicciones_norm = modelo(fechas_tensor, training=False).numpy().flatten()
            predicciones = predicciones_norm * (y_max - y_min) + y_min  # Desnormalizar
            
            return fechas, predicciones
        except Exception as e:
            st.error(f"Error en la predicción del rango: {str(e)}")
            return None, None
    else:
        st.warning("El modelo aún no está entrenado. Por favor, entrene el modelo primero.")
        return None, None

def predecir(fecha):
    if "entrenado" in st.session_state and st.session_state["entrenado"]:
        try:
            modelo = st.session_state["modelo"]
            
            # Asegurarse de que fecha sea un float
            fecha = float(fecha)
            
            # Normalizar la fecha con los valores guardados en sesión
            x_min = st.session_state["x_min"]
            x_max = st.session_state["x_max"]
            y_min = st.session_state["y_min"]
            y_max = st.session_state["y_max"]
            
            fecha_norm = (fecha - x_min) / (x_max - x_min)
            
            # Crear un tensor explícitamente con la forma correcta para TensorFlow
            fecha_tensor = tf.convert_to_tensor([[fecha_norm]], dtype=tf.float32)
            
            # Realizar la predicción con el tensor correcto
            prediccion_norm = modelo(fecha_tensor, training=False)
            
            # Convertir el tensor a un valor numérico
            prediccion_norm_value = prediccion_norm.numpy()[0][0]
            
            # Desnormalizar la predicción
            prediccion_real = prediccion_norm_value * (y_max - y_min) + y_min
            
            return prediccion_real
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
            import traceback
            st.error(f"Detalles: {traceback.format_exc()}")
            return None
    else:
        st.warning("El modelo aún no está entrenado. Por favor, entrene el modelo primero.")
        return None
        
def verificar_entrenamiento():
    return st.session_state.get("entrenado", False)

# Función para clasificar una imagen MNIST
def clasificar_mnist(imagen):
    if "mnist_entrenado" in st.session_state and st.session_state["mnist_entrenado"]:
        try:
            modelo = st.session_state["modelo_mnist"]
            
            # Normalizar imagen (asegurarse que esté entre 0-1)
            imagen_norm = imagen / 255.0
            
            # Realizar predicción
            pred = modelo.predict(imagen_norm.reshape(1, -1))
            
            # Obtener la clase con mayor probabilidad
            clase_pred = np.argmax(pred, axis=1)[0]
            
            return clase_pred, pred[0]
        except Exception as e:
            st.error(f"Error en la clasificación: {str(e)}")
            return None, None
    else:
        st.warning("El modelo MNIST aún no está entrenado.")
        return None, None


# Función para evaluar modelo MNIST con datos de prueba
def evaluar_modelo_mnist(archivo_test):
    if "mnist_entrenado" not in st.session_state or not st.session_state["mnist_entrenado"]:
        st.warning("El modelo MNIST aún no está entrenado. Por favor, entrena primero el modelo.")
        return
    
    try:
        # Cargar datos de prueba
        test_data = pd.read_csv(archivo_test, header=None)
        
        # Separar etiquetas y características
        y_test = test_data.iloc[:, 0].values  # Primera columna son las etiquetas
        X_test = test_data.iloc[:, 1:].values / 255.0  # Resto son píxeles, normalizados a [0,1]
        
        # Obtener el modelo entrenado
        modelo = st.session_state["modelo_mnist"]
        
        # Realizar predicciones
        predicciones = modelo.predict(X_test)
        clases_predichas = np.argmax(predicciones, axis=1)
        
        # Calcular precisión
        precision = np.mean(clases_predichas == y_test)
        st.success(f"Precisión del modelo en datos de prueba: {precision*100:.2f}%")
        
        # Crear tabla de comparación
        st.subheader("Comparación de predicciones")
        
        # Determinar número de ejemplos a mostrar (máximo 20 para no sobrecargar la UI)
        num_ejemplos = min(20, len(y_test))
        indices = np.random.choice(len(y_test), num_ejemplos, replace=False)
        
        # Crear columnas para mostrar tabla
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown("### Imagen")
        with col2:
            st.markdown("### Valor Real")
        with col3:
            st.markdown("### Predicción del Modelo")
        
        # Mostrar resultados
        for idx in indices:
            # Crear una fila para cada ejemplo
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                # Mostrar imagen
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                # Mostrar valor real
                st.markdown(f"<h2 style='text-align: center;'>{y_test[idx]}</h2>", unsafe_allow_html=True)
            
            with col3:
                # Mostrar predicción y confianza
                prediccion = clases_predichas[idx]
                confianza = predicciones[idx][prediccion] * 100
                
                # Color basado en si la predicción es correcta
                color = "green" if prediccion == y_test[idx] else "red"
                
                st.markdown(f"""
                <div style='text-align: left;'>
                    <h2 style='color: {color};'>{prediccion}</h2>
                    <p>Confianza: {confianza:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Agregar separador
            st.markdown("---")
        
        # Mostrar matriz de confusión
        st.subheader("Distribución de Predicciones")
        
        # Crear figura con dos subplots: distribución real y predicha
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribución real
        ax1.hist(y_test, bins=np.arange(11)-0.5, alpha=0.7)
        ax1.set_title("Distribución de valores reales")
        ax1.set_xticks(range(10))
        ax1.set_xlabel("Dígito")
        ax1.set_ylabel("Frecuencia")
        
        # Distribución predicha
        ax2.hist(clases_predichas, bins=np.arange(11)-0.5, alpha=0.7, color='orange')
        ax2.set_title("Distribución de predicciones")
        ax2.set_xticks(range(10))
        ax2.set_xlabel("Dígito")
        ax2.set_ylabel("Frecuencia")
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error al evaluar el modelo MNIST: {str(e)}")
        import traceback
        st.error(f"Detalles: {traceback.format_exc()}")


# Sidebar
st.sidebar.title("Menú")
trm_opcion = st.sidebar.checkbox("Predicción TRM")

if(trm_opcion):
    opcion = st.sidebar.radio("Selecciona una opción", ["Ninguno", "Entrenar Modelo", "Predecir Dólar"], disabled=False)
else:
     opcion = "Ninguno"
    

mnist_opcion = st.sidebar.checkbox("MNIST")
if(mnist_opcion):
    opcion2 = st.sidebar.radio("Selecciona una opción MNIST", ["Ninguno", "Entrenar Modelo MNIST", "Clasificar MNIST"], disabled=False)
else:
    opcion2 = "Ninguno"


# Entrenamiento TRM
if opcion == "Entrenar Modelo":
    st.title("Entrenamiento del Modelo TRM")
    descargar_formato()
    archivo = st.file_uploader("Carga tu archivo CSV o Excel", type=["csv", "xlsx"])

    
    if archivo:
        data = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
        st.write("Vista previa de los datos:")

        col1, col2 = st.columns(2)
        col1.subheader("Tabla de datos")
        col1.data_editor(data, height=300)
        
        col2.subheader("Gráfica XvY")
        fig, ax = plt.subplots()
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1])
        ax.set_xlabel("Fechas")
        ax.set_ylabel("Valor Dolar (COP)")
        ax.set_title("Gráfica Datos TRM")
        col2.pyplot(fig) 

        # Pedir al usuario la configuración de la red
        num_layers = st.number_input("Ingrese el número de capas ocultas:", min_value=1, step=1)
        neuronas_por_capa = [
        st.number_input(f"Ingrese el número de neuronas para la capa {i+1}:", min_value=1, step=1)
        for i in range(num_layers)
        ]  

        if st.button("Entrenar Modelo"):
            entrenar_modelo(data)
            st.success("Entrenamiento finalizado.")

# Predicción TRM
elif opcion == "Predecir Dólar":
    st.title("Predicción del Dólar")

    if "entrenado" not in st.session_state or not st.session_state["entrenado"]:
        st.warning("El modelo de predicción aún no está entrenado. Por favor, entrena primero el modelo.")
    else:

        opcionPD = st.radio("Selecciona una opción", ["Fecha", "Rango de fechas"])

        if(opcionPD == "Fecha"):
            fecha = st.number_input("Ingresa una fecha en formato numérico (YYYYMMDD)", min_value=0.0, step=1.0)
            if st.button("Predecir"):
                if verificar_entrenamiento():
                    resultado = predecir(fecha)
                    st.write(f"Valor estimado del dólar: {float(resultado):.2f}")
                else:
                    st.write("El modelo aún no está entrenado...")
        elif (opcionPD == "Rango de fechas"):
            # Predicción por rango de fechas
            fecha_inicio = st.number_input("Fecha de inicio (YYYYMMDD):", min_value=0.0, step=1.0)
            fecha_fin = st.number_input("Fecha de fin (YYYYMMDD):", min_value=0.0, step=1.0)

            if fecha_inicio > fecha_fin:
                st.error("La fecha de inicio no puede ser mayor que la fecha de fin.")
            else:
                if st.button("Predecir Rango"):
                    if "entrenado" in st.session_state and st.session_state["entrenado"]:
                        fechas, valores = predecir_rango(fecha_inicio, fecha_fin)
                        if fechas is not None:
                            # Convertir fechas numéricas a formato de fecha real
                            fechas_formateadas = [datetime.datetime.strptime(str(int(f)), "%Y%m%d") for f in fechas]
                            
                            fig, ax = plt.subplots()
                            ax.plot(fechas_formateadas, valores, marker='o', linestyle='-', color='b', label='Predicción')
                            ax.set_xlabel("Fecha")
                            ax.set_ylabel("Valor Dólar (COP)")
                            ax.set_title("Predicción del Dólar en Rango de Fechas")
                            ax.legend()
                            
                            # Formatear eje X para mostrar fechas de forma legible
                            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                            plt.xticks(rotation=45)
                            
                            # Añadir etiquetas de valores sobre cada punto
                            for i, txt in enumerate(valores):
                                ax.annotate(f"{txt:.2f}", (fechas_formateadas[i], valores[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='black')
                            
                            st.pyplot(fig)
                    else:
                        st.warning("El modelo aún no está entrenado.")

# Entrenamiento MNIST
elif opcion2 == "Entrenar Modelo MNIST":
    st.title("Entrenamiento del Modelo MNIST")
    
    
    # Cargar archivo MNIST
    st.info("Carga tu archivo CSV con los datos MNIST. La primera columna debe contener las etiquetas (0-9) y las demás columnas los píxeles (784 columnas).")
    archivo_mnist = st.file_uploader("Carga tu archivo MNIST CSV", type=["csv"])
    
    if archivo_mnist:
        # Cargar y procesar datos
        X, y, y_one_hot = cargar_mnist(archivo_mnist)
        
        if X is not None:
            st.success(f"Datos cargados correctamente. {X.shape[0]} imágenes de {X.shape[1]} píxeles.")
            
            # Mostrar algunos ejemplos
            st.subheader("Vista previa de los datos")
            n_ejemplos = min(5, X.shape[0])
            fig, axs = plt.subplots(1, n_ejemplos, figsize=(12, 3))
            
            for i in range(n_ejemplos):
                axs[i].imshow(X[i].reshape(28, 28), cmap='gray')
                axs[i].set_title(f"Etiqueta: {y[i]}")
                axs[i].axis('off')
            
            st.pyplot(fig)
            
            # Configuración del modelo
            col1, col2 = st.columns(2)
            with col1:
                num_layers_mnist = st.number_input("Número de capas ocultas para MNIST:", min_value=1, max_value=5, value=2, step=1)
            
            neuronas_mnist = []
            for i in range(num_layers_mnist):
                with col2 if i % 2 else col1:
                    neuronas = st.number_input(f"Neuronas en capa {i+1}:", min_value=1, value=128 if i==0 else 64, step=10)
                    neuronas_mnist.append(neuronas)
            
            # Botón de entrenamiento
            if st.button("Entrenar Modelo MNIST"):
                modelo_mnist, history = entrenar_modelo_mnist(X, y_one_hot, num_layers_mnist, neuronas_mnist)
                

# Clasificación MNIST
elif opcion2 == "Clasificar MNIST":
    st.title("Clasificación de Dígitos MNIST")
    
    if "mnist_entrenado" not in st.session_state or not st.session_state["mnist_entrenado"]:
        st.warning("El modelo MNIST aún no está entrenado. Por favor, entrena primero el modelo.")
    else:
        st.success("Modelo MNIST cargado y listo para clasificar.")
        
        # Opciones de clasificación
        modo_clasificacion = st.radio(
            "Selecciona el modo de clasificación:",
            ["Evaluar con archivo de prueba"]
        )
        
        if modo_clasificacion == "Evaluar con archivo de prueba":
            # Cargar archivo de prueba MNIST
            st.info("Carga tu archivo CSV de prueba MNIST. Debe tener el mismo formato que el archivo de entrenamiento.")
            archivo_test = st.file_uploader("Carga tu archivo MNIST de prueba", type=["csv"])
            
            if archivo_test:
                evaluar_modelo_mnist(archivo_test)
        
       