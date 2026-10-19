import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

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
        # Measure current states. The 0 selects which robot in the simulation to read
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command q_d means Desired joint position, and qd_d means Desired joint velocity
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)  # Set the torque command
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

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
        # qdd_mes represent the acceleration of last step.
        # qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        # Y describes how the robot's mass and inertia produce joint torques.
        Y_step = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
        # Save a copy because the next calculation may reuse the same matrix. Y_step.copy() is for put the only the value into the container. If using Y_step directly, the next step will overwrite the previous one.
        regressor_all.append(Y_step.copy())
        # Put torques in the same joint order as the rows of Y_step.
        tau_mes_all.append(dyn_model.ReoderJoints2PinVec(tau_mes, "vel").copy())
        
        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torque and compute the parameters 'a'  using pseudoinverse for all the joint
    # Join all small matrices into one tall matrix, with one row per joint per step.
    Y = np.vstack(regressor_all)
    # Join the torque readings into a vector with the same row order.
    tau = np.concatenate(tau_mes_all)
    # Solve tau = Y @ a; pinv finds the parameters with the smallest squared error.
    a = np.linalg.pinv(Y) @ tau
    # Use the estimated parameters to predict the recorded torques.
    tau_pred = Y @ a
    print("All-joint inertial parameters a:", a)
    
    # TODO reshape the regressor and the torque vector to isolate the last joint and find the its dynamical parameters
    # Reshape into [time steps, joints, parameters], then select the last joint.
    # Only the final link's 10 mass and inertia parameters affect its torque. Become a 3D matrix [time, joints, parameters]
    Y_last = Y.reshape(-1, num_joints, Y.shape[1])[:, -1, -10:]
    # Y_grouped = Y.reshape(-1, num_joints, Y.shape[1])
    # Y_last = Y_grouped[:, -1, -10:]
    tau_last = tau.reshape(-1, num_joints)[:, -1]
    a_last = np.linalg.pinv(Y_last) @ tau_last
    print("Last-joint inertial parameters:", a_last)
    
    # TODO compute the metrics (R-squared adjusted etc...) for the linear model on a different file 
    # Check both models using the same samples used to fit them.
    for label, matrix, measured, predicted in (
        ("All joints", Y, tau, tau_pred),
        ("Last joint", Y_last, tau_last, Y_last @ a_last),
    ):
        # RMSE and MAE measure prediction error; smaller values are better.
        residual = measured - predicted
        rmse = np.sqrt(np.mean(residual ** 2))
        mae = np.mean(np.abs(residual))
        # Compare squared prediction errors with the torque's variation around its mean.
        sse = np.sum(residual ** 2)
        sst = np.sum((measured - measured.mean()) ** 2)
        # R-squared near 1 means a good fit; it is undefined for constant torque.
        r_squared = 1 - sse / sst if sst > 0 else np.nan
        # Rank counts independent parameters because some columns contain repeated information.
        rank = np.linalg.matrix_rank(matrix)
        n = measured.size
        # Adjusted R-squared accounts for model size; no extra intercept is fitted here.
        adjusted = 1 - (1 - r_squared) * (n - 1) / (n - rank) if n > rank else np.nan
        print(f"{label}: RMSE={rmse:.6g}, MAE={mae:.6g}, "
              f"R-squared={r_squared:.6g}, adjusted R-squared={adjusted:.6g}")
   
    # TODO plot the torque prediction error for each joint (optional)
    # A prediction error is measured torque minus predicted torque.
    error = (tau - tau_pred).reshape(-1, num_joints)
    # Convert sample numbers into seconds using the simulation time step.
    times = np.arange(1, len(error) + 1) * time_step
    # Create one plot per joint so each joint's errors are easy to see.
    fig, axes = plt.subplots(num_joints, 1, sharex=True, figsize=(10, 2 * num_joints))
    for joint, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(times, error[:, joint])
        ax.set_ylabel(f"Joint {joint + 1}\nerror (N m)")
        ax.grid(True)
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
