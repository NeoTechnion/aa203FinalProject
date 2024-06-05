from isaacgym import gymapi, gymutil  # Import isaacgym first
import torch  # Import torch after isaacgym

def main():
    try:
        # Initialize gym
        gym = gymapi.acquire_gym()

        # Parse arguments
        args = gymutil.parse_arguments(description="URDF Loading Example")

        # Configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True

        # Create sim
        sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            raise Exception("Failed to create sim")

        # Create viewer
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise Exception("Failed to create viewer")

        # Load asset
        asset_root = "/home/yimeng/DTU-RL-Isaac-Gym-Drone-Env/assets/urdf"
        urdf_file = "Drone.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        # Ensure the asset path is correct
        try:
            drone_asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
        except Exception as e:
            raise Exception(f"Failed to load asset: {e}")

        # Create environment
        env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, -1.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

        # Add drone asset to environment
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
        drone_handle = gym.create_actor(env, drone_asset, pose, "drone", 0, 1)

        # Main simulation loop
        while not gym.query_viewer_has_closed(viewer):
            # Step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Update the viewer
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

        # Cleanup
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
