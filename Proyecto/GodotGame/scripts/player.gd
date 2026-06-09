extends Node2D

var health: int = 5
@onready var health_label = $HealthLabel
@onready var body = $Body

func _ready():
    update_ui()

func take_damage(amount: int):
    health -= amount
    update_ui()
    # Efecto de daño
    var tween = create_tween()
    body.color = Color.RED
    tween.tween_property(body, "color", Color.GREEN_YELLOW, 0.2)
    
    if health <= 0:
        print("GAME OVER")

func update_ui():
    health_label.text = "HP: " + str(health)
