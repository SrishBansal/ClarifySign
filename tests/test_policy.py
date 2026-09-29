from core.clarification import Planner
def test_ambiguous(): assert Planner().decide([("blue",.49),("black",.39),("green",.12)])['action']=="clarify"
def test_clear(): assert Planner().decide([("shirt",.91),("shoe",.06),("trouser",.03)])['action']=="commit"
