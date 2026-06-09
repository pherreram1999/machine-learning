extends Node2D

signal attack_player(damage: int)
signal defeated(enemy_ref)

var time_left: float = 10.0
var max_time: float = 10.0
var required_shapes: Array = []
var active: bool = false
var in_trench: bool = true

@onready var time_bar = $TimeBar
@onready var symbol_label = $Card/SymbolText

func setup(shapes: Array, time: float):
    required_shapes = shapes.duplicate()
    max_time = time
    time_left = max_time
    update_card_ui()
    time_bar.max_value = max_time
    time_bar.value = time_left

func _process(delta):
    if not active: return
    
    time_left -= delta
    time_bar.value = time_left
    if time_left <= 0:
        attack()

func appear(target_x: float, target_y: float):
    in_trench = false
    position.x = target_x
    position.y = target_y + 300 # Empieza en la trinchera abajo
    
    var tween = create_tween()
    tween.tween_property(self, "position:y", target_y, 0.5).set_trans(Tween.TRANS_SINE)
    tween.tween_callback(func(): active = true)

func attack():
    active = false
    attack_player.emit(1)
    retreat()

func try_damage(shape: String) -> bool:
    if not active: return false
    
    if required_shapes.size() > 0 and required_shapes[0] == shape:
        required_shapes.pop_front()
        update_card_ui()
        if required_shapes.is_empty():
            die()
        return true
    return false

func update_card_ui():
    if required_shapes.is_empty():
        symbol_label.text = ""
    else:
        var txt = ""
        for s in required_shapes:
            txt += s + "\n"
        symbol_label.text = txt.strip_edges()

func die():
    active = false
    defeated.emit(self)
    retreat()

func retreat():
    in_trench = true
    var tween = create_tween()
    tween.tween_property(self, "position:y", position.y + 300, 0.5).set_trans(Tween.TRANS_SINE)
    tween.tween_callback(queue_free)
