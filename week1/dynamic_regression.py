import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from regression_metrics import regression_metrics

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the sibling RoboEnv assets when the lab has no local configuration.
    if not os.path.isfile(os.path.join(cur_dir, "configs", conf_file_name)):
        cur_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "RoboEnv"))
    headless = os.environ.get("ROBOENV_HEADLESS") == "1"
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque com
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        # Pair the pre-step state with acceleration and torque from that step.
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        
        # TODO Compute regressor and store it
        regressor_all.append(dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes).copy())
        # The regressor rows and stored torques must use Pinocchio joint order.
        tau_mes_all.append(dyn_model.ReoderJoints2PinVec(tau_mes, "vel").copy())
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    if not regressor_all:
        print("No samples collected; identification skipped.")
        return
    # Stack one row per joint per sample: tau = Y @ a.
    Y = np.vstack(regressor_all)
    tau = np.concatenate(tau_mes_all)
    a = np.linalg.pinv(Y) @ tau
    tau_pred = Y @ a
    print("All-joint inertial parameters a:", a)
    
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    # Each moving link contributes 10 columns; the final joint uses the final link.
    Y_last = Y.reshape(-1, num_joints, Y.shape[1])[:, -1, -10:]
    tau_last = tau.reshape(-1, num_joints)[:, -1]
    a_last = np.linalg.pinv(Y_last) @ tau_last
    print("Last-joint inertial parameters:", a_last)
    
    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    # These metrics describe the fit on the collected training samples.
    print("All-joint metrics:", regression_metrics(tau, tau_pred, Y))
    print("Last-joint metrics:", regression_metrics(tau_last, Y_last @ a_last, Y_last))
   
    # TODO plot the torque prediction error for each joint (optional)
    # Show measured minus predicted torque, one subplot per joint.
    error = (tau - tau_pred).reshape(-1, num_joints)
    times = np.arange(1, len(error) + 1) * time_step
    fig, axes = plt.subplots(num_joints, 1, sharex=True, figsize=(10, 2 * num_joints))
    for joint, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(times, error[:, joint])
        ax.set_ylabel(f"Joint {joint + 1}\nerror (N m)")
        ax.grid(True)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    if headless:
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    main()
