extends Control

@onready var high_score_label = $HighScoreLabel

func _ready():
    high_score_label.text = "High Score: " + str(Global.high_score)

func _on_play_button_pressed():
    get_tree().change_scene_to_file("res://Main.tscn")
