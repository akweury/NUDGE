
## Reasoner

### Inputs:
- ``

### Output
- `decision: kill / avoid / align`

### Pseudo Code:
```python

def collide(player, enemy, k=10):
    raise NotImplementedError


def stop_moving(player):
    # check if player moved in past k frames
    raise NotImplementedError


def aligned(player, target_obj):
    raise NotImplementedError


# pathfinder
def find_target_object(next_obj, target_obj, sub_target_obj, player):
    if next_obj is None:
        next_obj = target_obj
    if stop_moving(player):
        if aligned(player, next_obj):
            next_obj = target_obj
        else:
            next_obj = sub_target_obj
    return next_obj


# decision for the enemy: kill, avoid, or ignore

def decision_maker(player, enemy, next_obj):
    # check with enemy first (mask)
    # invent predicate like : killable, exist, collide
    # killable: player fired then it vanished 
        
    if enemy is not None:
        if collide(player, enemy):
            if enemy.killable:
                # if enemy exist, enemy killable, enemy collide => kill
                # clause:    kill := exist(enemy), killable(enemy), collide(enemy, agent)
                return "kill", enemy
            else:
                # if enemy exist, enemy not killable, enemy collide => avoid
                # clause:    avoid := exist(enemy), not_killable(enemy), collide(enemy, agent)
                return "avoid", enemy
    # if no enemy exist, align to target object
    # clause:    align := not_exist(enemy), align(next_obj)
    # clause:    align := exist(enemy), not_collide(enemy, agent).
    # ignore enemy and align to the target object 
    return "align", next_obj


def reasoner(next_obj, target_obj, sub_target_obj, player):
    next_obj, enemy_obj = find_target_object(next_obj, target_obj, sub_target_obj, player)
    decision, decision_obj = decision_maker(player, enemy_obj, next_obj)
    return decision, decision_obj


player = None
decision_obj = None
target_obj = 1  # given 
sub_target_obj = 2  # learned
while True:
    # the reasoner has to know, player, target objects, sub_target_objects
    decision, decision_obj = reasoner(decision_obj, target_obj, sub_target_obj, player)
    print(decision)  # Output: kill/avoid/align

```





