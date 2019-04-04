import pickle
import sqlite3
import pandas as pd
from numpy import dtype as ndtype

class DatosSQL(object):
    """
    Objeto que agrupa las métodos guardar, tabla_df y id_df
    los cuales trabajan sobre la base de datos.
    """
    def __init__(self, path_db):
        self.path_db = path_db

    def __str__(self):
        return f'Datos de {self.path_db}'

    def __repr__(self):
        return f'<object DatosSQL de {self.path_db} en {hex(id(self))}>'

    sqlinsertar = "INSERT INTO {tabla} {keys} VALUES {num_q}"
    sqlalterar = "ALTER TABLE {tabla} ADD COLUMN {nuevo} {tipo}"
    sqltodo = "SELECT * FROM {tabla}"
    sqlgetid = "SELECT * FROM {tabla} WHERE id={id}"
    sqltablas = 'SELECT name FROM sqlite_master WHERE type = "table"'
    sqlids = "SELECT id FROM {tabla}"
    sqlcolumnas = "PRAGMA table_info({tabla})"

    def _conexion(self):
        """
        Devuelve los objetos cursor y conección
        tras haberse conectado con la base de datos.
        """
        conn = sqlite3.connect(self.path_db)
        c = conn.cursor()
        return c, conn

    #Funciones para inserción de datos
    def _cachar_tabla(self, diccionario):
        """
        Devuelve el nombre de la tabla a partir de la
        presencia de keys únicas a cada tabla, las que
        deberían estar en el diccionario. En caso contrario,
        arroja un error.
        """
        if 'normalizar_corromper' in diccionario.keys():
            return 'estructura'
        elif 'infodae' in diccionario.keys():
            return 'superestructura'
        elif 'estructura' in diccionario.keys():
            return 'infodae'
        elif 'superestructura' in diccionario.keys():
            return 'infolstm'
        else:
            raise ValueError(
                f'{self.diccionario} no contiene los parámetros de una tabla en la base de datos.'
            )
        #if 'a' in diccionario.keys():
        #    return 'hola'
        #elif 'modeloDAE' in diccionario.keys():
        #    return 'modelo'
        #elif 'Loss_valid' in diccionario.keys():
        #    return 'pick'
        #else:
        #    raise ValueError(f'{self.diccionario} no contiene los parámetros de una tabla en la base de datos.')
    
    def _obtener_rows(self, diccionario):
        """
        Devuelve los nombres de las columnas y la tabla
        donde se insertarán los nuevos datos.
        """
        tabla = self._cachar_tabla(diccionario)
        rows = self._obtener_datos(self.sqlcolumnas.format(tabla=tabla), 1)
        rows.remove('id')
        return rows, tabla


    def _tipo_de_valor(self, valores):
        """
        Devuelve un diccionario con el nombre de las
        llaves nuevas del diccionario y tipo de data
        que le correspondería en SQLite.
        """
        valor_tipo = {}
        for llave, valor in valores.items():
            if valor is int:
                valor_tipo[llave] = 'INTEGER'
            elif valor is float:
                valor_tipo[llave] = 'REAL'
            elif valor is str:
                valor_tipo[llave] = 'TEXT'
            elif valor is bytes:
                valor_tipo[llave] = 'BLOB'
            else:
                raise ValueError(
                    f'(El diccionario contiene un valor no admitido en SQLite3 ({llave}))'
                )
        return valor_tipo

    def _valores_nuevos(self, llaves, diccionario):
        """
        Crea un dict con los valores nuevos presentes
        en el diccionario asociados al tipo de valor
        que estos contienen.
        Devuelve el resultado de _tipo_de_valor.
        """
        valores = {}
        for llave, valor in diccionario.items():
            if llave not in llaves:
                valores[llave] = type(valor)
        valor_tipo = self._tipo_de_valor(valores)
        return valor_tipo 

    def _agregar_columnas(self, tabla, llaves, diccionario):
        """
        Altera la tabla correspondiente con las nuevas columnas.
        """
        c, conn = self._conexion()
        nuevos = self._valores_nuevos(llaves, diccionario)
        for nuevo, tipo in nuevos.items():
            c.execute(
                self.sqlalterar.format(tabla=tabla, nuevo=nuevo, tipo=tipo)
            )
            conn.commit()
        c.close()
        conn.close()

    def _get_preg(self, diccionario):
        """
        Crea el string de una tupla de signos de
        interrogración para integrar en la query.
        (i.e. '(?, ?, ?)')
        """
        num = '?,' * len(diccionario.keys())
        num = num[:-1]
        num_q = f'({num})'
        return num_q

    def _get_keys(self, diccionario):
        """
        Devuelve una tupla con las llaves del diccionario
        o, en caso de ser una llave única, una string de una
        tupla sin la trailing comma.
        """
        if len(diccionario.keys()) > 1:
            return tuple(diccionario.keys())
        else: 
            return str(list(diccionario.keys())).replace(']', ')').replace('[', '(')

    def _insertar_valores(self, tabla, diccionario):
        """
        Inserta los datos en la base de datos.
        Devuelve el id del nuevo row creado.
        """
        c, conn = self._conexion()
        num_q = self._get_preg(diccionario)
        keys = self._get_keys(diccionario)
        c.execute(
            self.sqlinsertar.format(tabla=tabla, keys=keys, num_q=num_q),
            tuple(diccionario.values())
        )
        conn.commit()
        pk = c.lastrowid
        c.close()
        conn.close()
        return pk

    def guardar(self, diccionario):
        """
        Si las keys del diccionario son identicas a las columnas
        de la base de datos, los valores se insertan inmediatamente. 
        Si hubieran keys nuevos, la tabla se altera usando la función _agregar_columnas.
        """
        llaves, tabla = self._obtener_rows(diccionario)
        if list(diccionario.keys()) == llaves:
            return self._insertar_valores(tabla, diccionario)
        else:
            self._agregar_columnas(tabla, llaves, diccionario)
            return self._insertar_valores(tabla, diccionario)

    #Funciones para extracción de datos
    def _obtener_datos(self, query, indice):
        """
        Obtiene los datos de una query, devolviéndolos
        como una lista.
        """
        c, conn = self._conexion()
        c.execute(query)
        tablas = c.fetchall()
        c.close()
        conn.close()
        tablas = [x[indice] for x in tablas]
        return tablas

    def _tabla_en_db(self, tabla):
        """
        Confirma la existencia de la tabla en la base de datos.
        """
        tablas = self._obtener_datos(self.sqltablas, 0)
        if tabla in tablas:
            self.tabla = tabla
            return True
        else:
            raise ValueError(
                "La tabla ingresada no existe en la base de datos."
            )

    def _id_en_tabla(self, id):
        """
        Confirma la existencia de un ID en la tabla.
        """
        ides = self._obtener_datos(
            self.sqlids.format(tabla=self.tabla), 0
        )
        if id in ides:
            self.id = id
            return True
        else:
            raise ValueError(
                "No existe una instancia con ese ID en la tabla."
            )
    
    def _unpickle(self, df):
        """
        Devuelve el dataframe con los objetos (i.e. los blobs
        en la base de datos) reconvertidos a su formato original
        """
        unpickle = lambda x: pickle.loads(x) if type(x) is bytes else x
        for x in df.columns:
            if df[x].dtype is ndtype('O'):
                df[x] = df[x].map(unpickle, na_action='ignore')
        return df

    def tabla_df(self, tabla):
        """
        Devuelve un Pandas DataFrame con todos los datos
        de una tabla.
        """
        if self._tabla_en_db(tabla):
            c, conn = self._conexion()
            df = pd.read_sql_query(
                self.sqltodo.format(tabla=tabla),
                conn
            )
            c.close()
            conn.close()
            df = self._unpickle(df)
            return df

    def id_df(self, id, tabla):
        """
        Devuelve un Pandas DataFrame con los datos de un elemento
        de la tabla, elegido por el ID
        """
        if self._tabla_en_db(tabla) and self._id_en_tabla(id): 
            c, conn = self._conexion()
            un = pd.read_sql_query(
                self.sqlgetid.format(tabla=self.tabla, id=self.id),
                conn
            )
            c.close()
            conn.close()
            un = self._unpickle(un)
            return un

    def tablas(self):
        """
        Devuelve las tablas de la base de datos.
        """
        return self._obtener_datos(self.sqltablas, 0)


    def columnas(self, tabla):
        """
        Devuelve una lista con las columnas de la tabla.

        """
        return self._obtener_datos(
            self.sqlcolumnas.format(tabla=tabla),
            1
        )
