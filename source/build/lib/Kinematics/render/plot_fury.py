# This whole module is a quick implementation of 3d visualization.
import numpy as np
from fury import window, actor
import json
import os


def show_brick_with_json_file(elems, contps, solution, data_dir, factor=1e4):
    disps = [1 * i for i in solution['displacements']]
    nb_contps = len(contps)
    start = 0
    brick_actors = []

    for i, element in enumerate(elems.values()):
        center = np.array(
            [element.center[0], element.center[1], element.center[2]])

        #brick_centers[i] = center
        with open(os.path.join(data_dir, element.shape_file), 'r') as f:
            shape = json.load(f)
        shape = np.asarray(shape)
        color = np.random.rand(1, 3)
        # add initial brick
        brick_actor_init = actor.box(centers=center.reshape(1, 3),
                                     scales=shape, colors=color)
        brick_actor_init.GetProperty().SetOpacity(0.0)
        brick_actors.append(brick_actor_init)

        # add deformed brick
        brick_actor_def = actor.box(centers=np.zeros((1, 3)),
                                    scales=shape, colors=color)
        brick_actor_def.GetProperty().SetOpacity(1)
        brick_actor_def.SetOrientation(
            -factor*np.array([disps[start+i*6+3], disps[start+i*6+4], disps[start+i*6+5]])*180/np.pi)
        brick_actor_def.SetPosition(
            center+factor*np.array(disps[start+i*6:start+i*6+3]))
        brick_actors.append(brick_actor_def)
    # define a scene and add actors to it
    scene_failure = window.Scene()
    for ba in brick_actors:
        scene_failure.add(ba)
    scene_failure.add(actor.axes())
    # Create show manager.
    scene_failure.set_camera(position=(-200.0, -200, 200.0),
                             focal_point=(-50.0, 5, 40.0),
                             view_up=(-0.0, 0.0, 1.0))
    window.show(scene_failure)


def show_element_displaced(solution, data_dir):
    disps = [1 * i for i in solution['displacements']]
    # _______________________________________________________________________________-visualize

    # define a base as ground
    # geometry
    base_size = np.array([5, 5, 0.2])
    base_color = np.array([1, 1, 1])
    base_position = np.array([0, 0, -0.1])
    base_orientation = np.array([0, 0, 0, 1])
    # body
    base_actor = actor.box(centers=np.array([[0, 0, 0]]),
                           directions=[0, 0, 0],
                           scales=base_size,
                           colors=base_color)

    # render the bricks. All the bricks are rendered by a single actor for better performance.
    nb_bricks = 32
    brick_centers = np.zeros((nb_bricks, 3))

    brick_directions = np.zeros((nb_bricks, 3))
    brick_directions[:] = np.array([1.57, 0, 0])  # ? why 1.57
    #brick_orientations = np.zeros((nb_bricks, 3))

    brick_orns = np.zeros((nb_bricks, 4))

    brick_sizes = np.zeros((nb_bricks, 3))
    brick_colors = np.random.rand(nb_bricks, 3)
    # We use this array to store the reference of brick objects in pybullet world.
    bricks = np.zeros(nb_bricks, dtype=np.int8)
    # Logic to position the bricks appropriately to form a wall.
    geometry_path = "./process_data/"
    brick_actors = []
    element_dirs = (x[0] for x in os.walk(data_dir))
    # print(element_dirs)
    start = -6
    for i, element in enumerate(element_dirs):
        # print(element)
        if i == 0:
            continue
        geo_file = os.path.join(element, "geometry.txt")
        brick_file = os.path.join(element, "property.txt")
    #     # load geometry and property file
    #     with open(geo_file, 'r') as f_geo:
    #         with open(property_file, 'r') as f_property:
    # for i in range(nb_bricks):
    #     brick_file = os.path.join(geometry_path, f"element_{i}/property.txt")
        # read json file
        with open(brick_file, 'r') as f:
            property = json.load(f)
            center = np.asarray(property['center'])

            #brick_centers[i] = center
            shape = np.asarray(property['shape'])
            color = np.random.rand(1, 3)
            # add initial brick
            brick_actor_init = actor.box(centers=center.reshape(1, 3),
                                         scales=shape, colors=color)
            brick_actor_init.GetProperty().SetOpacity(0.5)
            # brick_actors.append(brick_actor_init)

            # add displaced brick

            orientation_rad = np.asarray(
                [disps[start+i*6+3], disps[start+i*6+4]*1, disps[start+i*6+5]])
            # print(orientation_rad[1])
            # print(orientation_rad)
            orientation_deg = orientation_rad*180/np.pi
            # brick_orientations[i] = orientation_deg
            #brick_directions[i] = transform.cart2sphere(disps[start+i*6+3],disps[start+i*6+4],disps[start+i*6+5])
            brick_actor = actor.box(centers=np.zeros((1, 3)),
                                    scales=shape, colors=color)
            # brick_actor.SetPosition(-center)
            brick_actor = actor.box(centers=np.zeros((1, 3)),
                                    scales=shape, colors=color)
            brick_actor.GetProperty().SetOpacity(1)
            brick_actor.SetOrientation(
                disps[start+i*6+3], disps[start+i*6+4], disps[start+i*6+5])
            brick_actor.SetPosition(center+disps[start+i*6:start+i*6+3])
            brick_actors.append(brick_actor)
            print(disps[start+i*6])

    # define a scene and add actors to it
    scene = window.Scene()
    scene.add(base_actor)
    for ba in brick_actors:
        scene.add(ba)

    # Create show manager.
    showm = window.ShowManager(scene, size=(900, 768), reset_camera=False,
                               order_transparent=True)

    showm.initialize()
    # camera
    scene.set_camera(position=(10.46, -8.13, 6.18),
                     focal_point=(0.0, 0.0, 0.79),
                     view_up=(-0.27, 0.26, 0.90))
    showm.render()
    showm.start()
