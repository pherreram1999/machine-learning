extends Control

@onready var score_label = $ScoreLabel
@onready var high_score_label = $HighScoreLabel
@onready var new_record_label = $NewRecordLabel

func _ready():
    score_label.text = "Score: " + str(Global.current_score)
    high_score_label.text = "High Score: " + str(Global.high_score)
    
    # Comprobar si hubo un nuevo record
    if Global.current_score >= Global.high_score and Global.current_score > 0:
        new_record_label.visible = true
    else:
        new_record_label.visible = false

func _on_retry_button_pressed():
    get_tree().change_scene_to_file("res://Main.tscn")

func _on_menu_button_pressed():
    get_tree().change_scene_to_file("res://MainMenu.tscn")
