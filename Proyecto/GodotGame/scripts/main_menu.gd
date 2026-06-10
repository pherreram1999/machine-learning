extends Control

@onready var high_score_label = $HighScoreLabel
@onready var selection_panel = $SelectionPanel
@onready var loading_panel = $LoadingPanel

func _ready():
    high_score_label.text = "High Score: " + str(Global.high_score)

func _on_play_button_pressed():
    selection_panel.visible = true

func _on_btn_camara_pressed():
    start_game("camara")

func _on_btn_mouse_pressed():
    start_game("mouse")

func start_game(mode: String):
    Global.input_mode = mode
    selection_panel.visible = false
    loading_panel.visible = true
    
    # Launch python script
    var godot_dir = ProjectSettings.globalize_path("res://")
    var script_path = godot_dir + "/../captura_y_prediccion.py"
    
    var args = [script_path, "--modo", mode]
    # open_console = false so it runs in the background
    var pid = OS.create_process("python", args, false)
    
    if pid != -1:
        Global.python_pid = pid
        print("Python script started with PID: ", pid, " at path: ", script_path)
        # Esperar la señal de ready
        await UDPReceiver.python_ready
    else:
        print("Failed to start python script. Make sure python is in PATH.")
        
    loading_panel.visible = false
    get_tree().change_scene_to_file("res://Main.tscn")
