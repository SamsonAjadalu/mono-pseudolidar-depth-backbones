python
import glob
import os
import sys
import random
import time
import numpy as np
import cv2

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    actor_list = []
    
    try:
        # 1. Connect to Client
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # 2. Spawn Ego Vehicle
        bp = blueprint_library.filter('model3')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        print(f"Spawned ego vehicle: {vehicle.type_id}")

        # 3. Attach RGB Camera (Simulating KITTI viewpoint)
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1242') # KITTI width
        cam_bp.set_attribute('image_size_y', '375')  # KITTI height
        cam_bp.set_attribute('fov', '90')
        
        # Position: 1.6m high, slightly forward (typical dash/roof mount)
        spawn_point_cam = carla.Transform(carla.Location(x=1.0, z=1.6))
        sensor = world.spawn_actor(cam_bp, spawn_point_cam, attach_to=vehicle)
        actor_list.append(sensor)
        
        # 4. Define Callback to Save Data
        print("Recording data... Press Ctrl+C to stop.")
        # Create output dir if not exists
        if not os.path.exists('output_samples'):
            os.makedirs('output_samples')

        # Lambda to save images (just for demo)
        sensor.listen(lambda image: image.save_to_disk('output_samples/%06d.png' % image.frame))

        # Keep the script running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        print("Cleaning up actors...")
        for actor in actor_list:
            actor.destroy()
        print("Done.")

if __name__ == '__main__':
    main()
