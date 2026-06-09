extends Node

signal figura_reconocida(figura_nombre, confianza)

var udp_server := UDPServer.new()
var peer : PacketPeerUDP

func _ready():
    var puerto = 12345
    udp_server.listen(puerto)
    print("Godot escuchando predicciones UDP en el puerto: ", puerto)

func _process(_delta):
    udp_server.poll()
    if udp_server.is_connection_available():
        peer = udp_server.take_connection()
    
    if peer != null:
        while peer.get_available_packet_count() > 0:
            var paquete_bytes = peer.get_packet()
            var json_string = paquete_bytes.get_string_from_utf8()
            var json = JSON.new()
            var error = json.parse(json_string)
            if error == OK:
                var datos = json.data
                procesar_prediccion(datos)

func procesar_prediccion(datos: Dictionary):
    var figura_detectada = datos.get("figura", "Ninguna")
    var confianza = datos.get("confianza", 0.0)
    
    print("UDP Recibido -> Figura: ", figura_detectada, " | Confianza: ", confianza)
    if figura_detectada != "Ninguna":
        figura_reconocida.emit(figura_detectada, confianza)
