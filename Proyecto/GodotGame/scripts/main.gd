extends Node2D

@export var enemy_scene: PackedScene

var enemies_defeated = 0
var active_enemies = []
var max_enemies_on_screen = 3

# Distribución en 3 columnas en la parte superior, según el boceto
var spawn_positions = [Vector2(300, 200), Vector2(640, 200), Vector2(980, 200)]

# Figuras en base al nuevo entrenamiento del perceptrón
var shapes_basic = ["Circulo", "Triangulo", "Cuadrado"]
var shapes_advanced = ["Circulo", "Triangulo", "Cuadrado", "Infinito", "Letra V", "Letra Z", "Letra B", "Letra N"]

@onready var player = $Player
@onready var spawn_timer = $SpawnTimer
@onready var health_ui = $CanvasLayer/HealthUI
@onready var pause_panel = $CanvasLayer/PausePanel

var total_hp = 5
var heart_halves = [] # Para guardar referencia a cada rectangulo de vida

func _ready():
    # Conectamos con el Autoload UDPReceiver
    UDPReceiver.figura_reconocida.connect(_on_figura_reconocida)
    spawn_timer.timeout.connect(_on_spawn_timer_timeout)
    spawn_timer.start(3.0)
    _setup_health_ui()

func _on_spawn_timer_timeout():
    # Limpiamos referencias nulas
    active_enemies = active_enemies.filter(func(e): return is_instance_valid(e))
    
    if active_enemies.size() < max_enemies_on_screen:
        spawn_enemy()

func spawn_enemy():
    if not enemy_scene: 
        print("Error: Escena del Enemigo no asignada en Main.")
        return
    
    # DIFICULTAD PROGRESIVA
    # 1. Menos tiempo antes de atacar
    var time_to_defeat = 10.0 - (enemies_defeated * 0.15)
    if time_to_defeat < 4.0: time_to_defeat = 4.0
    
    # 2. Figuras más complejas
    var shapes_to_use = shapes_basic
    var amount_shapes = 1
    
    if enemies_defeated >= 5:
        shapes_to_use = shapes_advanced
    
    # 3. Múltiples figuras para derrotar
    if enemies_defeated >= 15:
        if randf() > 0.6: amount_shapes = 2
        
    var required_shapes = []
    for i in range(amount_shapes):
        required_shapes.append(shapes_to_use[randi() % shapes_to_use.size()])
        
    var spawn_pos = spawn_positions[randi() % spawn_positions.size()]
    # Evitar spawnear encima de un enemigo existente en la misma columna
    for e in active_enemies:
        if is_instance_valid(e) and abs(e.position.x - spawn_pos.x) < 10:
            return # Posición ocupada, saltamos el spawn
    
    var enemy = enemy_scene.instantiate()
    add_child(enemy)
    active_enemies.append(enemy)
    
    enemy.setup(required_shapes, time_to_defeat)
    enemy.appear(spawn_pos.x, spawn_pos.y)
    
    enemy.attack_player.connect(_on_enemy_attack)
    enemy.defeated.connect(_on_enemy_defeated)

func _on_enemy_attack(damage: int):
    player.take_damage(damage)
    _update_health_ui(player.health)
    
    if player.health <= 0:
        _game_over()

func _game_over():
    Global.save_score(enemies_defeated)
    get_tree().change_scene_to_file("res://GameOver.tscn")

func _setup_health_ui():
    # Borrar hijos previos por si acaso
    for child in health_ui.get_children():
        child.queue_free()
        
    heart_halves.clear()
    
    # Crear contenedores de corazon
    # Necesitamos (total_hp / 2) redondeado hacia arriba corazones.
    var corazones = ceil(total_hp / 2.0)
    
    for i in range(corazones):
        var heart_box = HBoxContainer.new()
        heart_box.add_theme_constant_override("separation", 2)
        
        var left_half = ColorRect.new()
        left_half.custom_minimum_size = Vector2(15, 30)
        left_half.color = Color.RED
        
        var right_half = ColorRect.new()
        right_half.custom_minimum_size = Vector2(15, 30)
        right_half.color = Color.RED
        
        heart_box.add_child(left_half)
        heart_box.add_child(right_half)
        
        health_ui.add_child(heart_box)
        heart_halves.append(left_half)
        heart_halves.append(right_half)
        
    _update_health_ui(total_hp)

func _update_health_ui(current_health: int):
    for i in range(heart_halves.size()):
        if i < current_health:
            heart_halves[i].color = Color.RED # Lleno
        else:
            heart_halves[i].color = Color(0.2, 0.0, 0.0, 1) # Vacio (rojo oscuro)

func _on_pause_button_pressed():
    get_tree().paused = true
    pause_panel.visible = true

func _on_resume_button_pressed():
    pause_panel.visible = false
    get_tree().paused = false

func _on_enemy_defeated(enemy_ref):
    enemies_defeated += 1
    print("Enemigo derrotado! Total: ", enemies_defeated)

func _on_figura_reconocida(figura_nombre: String, confianza: float):
    print("Main escuchó: ", figura_nombre, " con confianza ", confianza)
    # RESOLUCIÓN DE CONFLICTOS: Priorizar enemigo con menos tiempo
    var candidate = null
    
    for enemy in active_enemies:
        if is_instance_valid(enemy) and enemy.active:
            if enemy.required_shapes.size() > 0 and enemy.required_shapes[0] == figura_nombre:
                if candidate == null or enemy.time_left < candidate.time_left:
                    candidate = enemy
    
    if candidate != null:
        candidate.try_damage(figura_nombre)
