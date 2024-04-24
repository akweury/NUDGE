target_predicate = [
    'action:1:image',
]

neural_predicate = []

neural_predicate_2 = {
    'exist': 'exist:2:shape,image',
    'not_exist': 'not_exist:2:shape,image',
    'phi': 'phi:4:shape,player,phi,image',
    'rho': 'rho:4:shape,player,rho,image',
}

neural_predicate_3 = [
    'group_shape:2:group,group_shape',
]

const_dict = {
    'player': "amount_player",
    'image': 'target',
    'shape': 'enum',
    'group': 'amount_e',
    'phi': 'amount_phi',
    'rho': 'amount_rho',
}

attr_names = ['color', 'shape', 'rho', 'phi', 'group_shape', "slope", 'number']

color = ['pink', 'green', 'blue']
# shape = ['sphere', 'cube', 'cone', 'cylinder', 'line', 'circle', "conic"]

pred_obj_mapping = {
    'in': None,
    'shape_counter': ["sphere", "cube", "cone", "cylinder"],
    'color_counter': ["red", "green", "blue"],
    'shape': ['sphere', 'cube', 'cone', 'cylinder', 'line', 'circle', 'conic'],
    'color:': ['red', 'green', 'blue'],
    'phi': ['x', 'y', 'z'],
    'rho': ['x', 'y', 'z'],
    'slope': ['x', 'y', 'z'],
}

"""
(explaination)
line: average distance between any two objects next to each other, group size
or 
remember the distances as a list
"""

pred_pred_mapping = {
    'shape_counter': ['shape'],
    'color_counter': ['color'],
    'in': []
}
