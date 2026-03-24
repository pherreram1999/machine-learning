import polars as pl  # librería para manejo de dataframes (alternativa rápida a pandas)


class Dataset:
    """Clase para cargar y almacenar cualquier dataset desde un archivo CSV."""

    def __init__(
        self,
        ruta_csv: str,  # ruta al archivo CSV que contiene los datos
        columna_clase: str,  # nombre de la columna que contiene las etiquetas/clases
        columnas_excluir: list[str] | None = None,  # columnas a ignorar (como IDs)
    ):
        # leer el archivo CSV completo usando polars
        df = pl.read_csv(ruta_csv)
        # si hay columnas a excluir (por ejemplo "Id"), las eliminamos del dataframe
        if columnas_excluir:
            df = df.drop(columnas_excluir)
        # extraer la columna clase como una lista de strings
        self.clases: list[str] = df[columna_clase].cast(pl.Utf8).to_list()
        # obtener los nombres únicos de las clases, ordenados alfabéticamente
        self.nombres_clases: list[str] = sorted(set(self.clases))
        # eliminar la columna clase para quedarnos solo con las características numéricas
        df_caracteristicas = df.drop(columna_clase)
        # guardar los nombres de las columnas de características
        self.nombres_caracteristicas: list[str] = df_caracteristicas.columns
        # convertir cada fila a una lista de floats (cada fila es un punto en el espacio)
        self.caracteristicas: list[list[float]] = [
            [float(valor) for valor in fila]  # convertir cada valor de la tupla a float
            for fila in df_caracteristicas.rows()  # iterar sobre las filas del dataframe
        ]

    def __len__(self) -> int:
        """Retorna el número total de muestras en el dataset."""
        # la cantidad de muestras es igual a la cantidad de filas de características
        return len(self.caracteristicas)

    def info(self) -> None:
        """Imprime información general del dataset."""
        # mostrar cuántas muestras tiene el dataset
        print(f"Total de muestras: {len(self)}")
        # mostrar cuántas características tiene cada muestra
        print(f"Características por muestra: {len(self.caracteristicas[0])}")
        # mostrar los nombres de las características
        print(f"Características: {self.nombres_caracteristicas}")
        # mostrar las clases únicas encontradas
        print(f"Clases: {self.nombres_clases}")
