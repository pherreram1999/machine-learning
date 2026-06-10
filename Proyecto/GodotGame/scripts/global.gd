extends Node

var high_score: int = 0
var current_score: int = 0
var input_mode: String = "camara"
var python_pid: int = -1

func stop_python_script():
    if python_pid != -1:
        OS.kill(python_pid)
        print("Script de Python detenido (PID: ", python_pid, ")")
        python_pid = -1

func save_score(score: int) -> bool:
    current_score = score
    if score > high_score:
        high_score = score
        return true
    return false

func _notification(what):
    if what == NOTIFICATION_WM_CLOSE_REQUEST:
        stop_python_script()
