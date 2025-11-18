import pathlib

import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

assets_dir = pathlib.Path(__file__).parent.parent / 'assets'
robot_file = assets_dir / 'SO101/so101_new_calib.xml'
franka = scene.add_entity(
    gs.morphs.MJCF(file=robot_file.as_posix())
)

scene.build()

for i in range(1000):
    scene.step()
