extends Node

var high_score: int = 0
var current_score: int = 0

func save_score(score: int) -> bool:
    current_score = score
    if score > high_score:
        high_score = score
        return true
    return false
