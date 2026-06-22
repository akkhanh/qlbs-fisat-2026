PLAYER_ACTIONS = ("Move", "Talk", "Steal", "Attack", "Hide")
NPC_ACTIONS = ("Patrol", "Investigate", "Attack", "Flee", "Trust", "Ignore")

PLAYER_INDEX = {name: i for i, name in enumerate(PLAYER_ACTIONS)}
NPC_INDEX = {name: i for i, name in enumerate(NPC_ACTIONS)}

IMPORTANT_PLAYER_ACTIONS = frozenset(("Steal", "Attack", "Hide"))

APPROVED_RESPONSES = {
    "Move": frozenset(("Patrol", "Ignore")),
    "Talk": frozenset(("Trust", "Ignore")),
    "Steal": frozenset(("Investigate", "Attack")),
    "Attack": frozenset(("Attack", "Flee")),
    "Hide": frozenset(("Investigate", "Patrol")),
}
