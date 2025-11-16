import os
import pandas as pd
import matplotlib.pyplot as plt
import genesis as gs

def get_latest_model(log_dir,
                     run_name:str|None=None):

    # Check if there are any runs in the log directory
    if len(os.listdir(log_dir)) == 0:
        gs.logger.info("No runs found. Using random initialization")
        return None

    # Get the latest run
    latest_run = max([f for f in os.listdir(log_dir) if f != run_name],)
    gs.logger.info(f"Latest run is: {latest_run}")

    # Look for checkpoints folder
    checkpoints_dir = os.path.join(log_dir, latest_run, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        gs.logger.info("No checkpoints folder found. Using random initialization")
        return None

    # Get the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if len(checkpoints) == 0:
        gs.logger.info("No checkpoints found. Using random initialization")
        return None
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1]

    #Check for the model
    model_path = os.path.join(checkpoints_dir, latest_checkpoint, 'policy-model.pth')
    if not os.path.exists(model_path):
        gs.logger.info("No model found. Using random initialization")
        return None

    # If we are here, we have a model
    # Now copy the log.txt of the latest run to the new run
    if run_name is not None:
        log_file = os.path.join(log_dir, latest_run, 'log.txt')
        new_log_file = os.path.join(log_dir, run_name, 'log.txt')
        if not os.path.exists(new_log_file):
            gs.logger.info(f"Copying log file from {log_file} to {new_log_file}")
            with open(log_file, 'r') as f:
                with open(new_log_file, 'w') as g:
                    g.write(f.read())
        else:
            gs.logger.info(f"Log file {new_log_file} already exists")

    # TODO: Put hyperparameters in another file
    #       And also copy it between runs

    # Loading the latest model
    gs.logger.info(f"Loading model from checkpoint {latest_checkpoint} of run {latest_run}")
    return model_path

def show_reward_info(mean_reward, loss, reward_dict):
    gs.logger.info(f"REWARD {mean_reward:.6g},\tLOSS {loss:.6g}")
    # Log by pairs of keys
    keys = list(reward_dict.keys())
    max_odd_key_len = max(len(key) for key in keys[0::2])
    max_even_key_len = max(len(key) for key in keys[1::2])
    for i in range(0, len(keys), 2):
        key1 = keys[i]
        val1 = reward_dict[key1]
        line = f"{key1:<{max_odd_key_len+1}}: {val1:>12.6g}"
        key2 = keys[i+1]
        val2 = reward_dict[key2]
        line += f"\t\t{key2:<{max_even_key_len+1}}: {val2:>12.6g}"
        gs.logger.info(line)

def log_update(log_dir, training_run_name, update, mean_reward, loss, reward_dict_mean):
    with open(os.path.join(log_dir, training_run_name, 'log.txt'), 'a') as f:
        # If it is empty, write the header
        if os.path.getsize(os.path.join(log_dir, training_run_name, 'log.txt')) == 0:
            f.write("Training run name,Update,Mean reward,Loss")
            for key in reward_dict_mean:
                f.write(f",{key}")
            f.write("\n")
        f.write(f"{training_run_name},{update},{mean_reward},{loss}")
        for key in reward_dict_mean:
            f.write(f",{reward_dict_mean[key]}")
        f.write("\n")


def log_plot(log_dir, run_name):
    log_file = os.path.join(log_dir, run_name, 'log.txt')
    if not os.path.exists(log_file):
        gs.logger.warning(f"Error creating log plot: {log_file} not found")
        return

    df = pd.read_csv(log_file, header=0)
    updates = df['Update']
    rewards = df['Mean reward']
    losses = df['Loss']

    # Get the rest of the columns, whatever they are.
    rest_of_columns = [column for column in df.columns if not column in ['Training run name', 'Update', 'Mean reward', 'Loss']]
    reward_metric_columns = [column for column in rest_of_columns if not 'reward' in column.lower()]
    reward_component_columns = [column for column in rest_of_columns if 'reward' in column.lower()]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 14))
    fig.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,
        hspace=0.4
    )

    # Subplot 1: Reward and loss
    ax1.set_title("Reward and Loss")
    color = 'green'
    ax1.set_ylabel("Mean Reward", color=color)
    ax1.plot(updates, rewards, color=color, label="Mean Reward")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    color = 'red'
    ax1_twin.set_ylabel("Loss", color=color)
    ax1_twin.plot(updates, losses, color=color, label="Loss")
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1_twin.grid(True)

    # Subplot 2: Reward metrics
    ax2.set_title("Reward Metrics")
    ax2.set_xlabel("Update")
    axes_list = [ax2]  # Lista de ejes que iremos añadiendo con twin
    colors = ['cornflowerblue', 'coral', 'mediumseagreen', 'goldenrod', 'steelblue', 'sandybrown', 'slateblue', 'palevioletred', 'teal', 'orchid']
    if len(reward_metric_columns) > 0:
        # Graficamos la primera métrica en el eje principal
        m0 = reward_metric_columns[0]
        if m0 in df.columns:
            axes_list[0].plot(updates, df[m0], color=colors[0], label=m0)
            axes_list[0].set_ylabel(m0, color=colors[0])
            axes_list[0].tick_params(axis='y', labelcolor=colors[0])
        else:
            gs.logger.warning(f"Column '{m0}' not found in CSV")

        # Para las siguientes métricas, creamos un nuevo twin axis desplazado
        for i in range(1, len(reward_metric_columns)):
            metric = reward_metric_columns[i]
            if metric not in df.columns:
                gs.logger.warning(f"Column '{metric}' not found in CSV")
                continue

            ax_new = axes_list[0].twinx()
            # Desplaza el spine para que no se superpongan
            ax_new.spines.right.set_position(("axes", 1.0 + 0.085 * (i-1)))

            ax_new.plot(updates, df[metric], color=colors[i % len(colors)], label=metric)
            ax_new.set_ylabel(metric, color=colors[i % len(colors)])
            ax_new.tick_params(axis='y', labelcolor=colors[i % len(colors)])
            axes_list.append(ax_new)

    # -- Subplot 3: Contribuciones parciales a la recompensa (mismo eje Y). No twins --
    ax3.set_title("Reward Contributions")
    ax3.set_ylabel("Reward Component")
    ax3.set_xlabel("Update")
    for i, component in enumerate(reward_component_columns):
        if component not in df.columns:
            gs.logger.warning(f"Column '{component}' not found in CSV")
            continue
        ax3.plot(updates, df[component], color=colors[i % len(colors)], label=component)
    ax3.grid(True)

    plot_path = os.path.join(log_dir, run_name, 'log-plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    gs.logger.info(f"Log plot saved to {plot_path}")
